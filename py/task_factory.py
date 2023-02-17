import math, os, random, time, yaml, tqdm
import numpy as np
import torch
from torch import nn, optim, cuda
from torch.backends import cudnn
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import boxes
from dataset import parse_dataset_config, labels_to_class_weights, LoadImagesAndLabels
from utils import load_pretrained_torch_state_dict, load_pretrained_darknet_state_dict, \
    save_torch_state_dict, AverageMeter, ProgressMeter, plot_images, non_max_suppression, \
    clip_coords, xywh2xyxy, ap_per_class, load_classes
from model import Darknet, compute_loss
from wrapper import timer
from indicators import collect_depth, cal_depth_indicators
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
import absl.logging as log
log.set_verbosity(log.INFO)

Y_RANGE = 80

def get(task:str='train') -> Callable:
    """pass

    Args:
        pass

    Raises:
        pass

    Returns:
        pass
    """

    pool = dict(
        train  = Trainer,
        test   = Tester,
        detect = Detector
    )
    assert task in pool.keys(), "task {} does not exist !".format(task)

    return pool[task]

class BaseTask(ABC):
    """
    pass

    Args:
        Same as attributes

    Attributes:
        cfg (str):            *.cfg path
        imsz (int):           [min, max, test]
        weights (str):        initial weights path
        device (str):         device id (i.e. 0 or 0,1 or cpu)
        epochs (int): 
        batch-size (int):
        data (str):           *.data path
        ds(int)               pixel grid size
        multi_scale (bool):   adjust (67%% - 150%%) imsz every 10 batches
        rect (bool):          rectangular training
        cache_imgs (bool):    cache images for faster training
        weights (str):        initial weights path
        name (str):           renames results.txt to results_name.txt if supplied
        adam (bool):          use adam optimizer
        single_cls (bool):    train as single-class dataset
        freeze_layers (bool): freeze non-output layers
        conf_thres (float):   object confidence threshold
        iou_thres (float):    IOU threshold for NMS
        test_mode (str):      'test', 'study', 'benchmark'
        augment (bool):       augmented inference
        names (str):          *.names path'
        source (str):         valid file/folder, 0 for webcam
        output (str):         output folder
        fourcc (str):         output video codec (verify ffmpeg support)
        half (bool):          half precision FP16 inference
        view_img (bool):      display results
        save_txt (bool):      save results to *.txt'
        classes (int):        filter by class
        agnostic_nms(bool):   class-agnostic NMS
    """

    @abstractmethod
    def _load_options(self) -> Dict[str, Any]:
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """
        pass

    @abstractmethod
    def _build_dataset(self):
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """
        pass

    @abstractmethod
    def _build_model(self):
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """
        pass

    @abstractmethod
    def go(self):
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """
        pass

class Trainer(BaseTask):
    """
    pass

    Args:
       pass

    Attributes:
        pass

    """


    def __init__(self):
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """

        self._load_options('config.yaml')
        OPT['device'] = torch.device(OPT['device'])
        random.seed(OPT['seed'])
        np.random.seed(OPT['seed'])
        torch.manual_seed(OPT['seed'])
        torch.cuda.manual_seed_all(OPT['seed'])
        cudnn.benchmark = True
        self.scaler = cuda.amp.GradScaler()
        self.start_epoch = 0
        self.train_dataset, self.train_dataloader, self.val_dataloader, self.names, self.n_classes = \
            self._build_dataset()
        self.model, self.ema_model = self._build_model()
        self.optimizer = self.define_optimizer(self.model)
        log.info("Check whether to load pretrained model weights...")
        if OPT['pretrained'].endswith(".pth.tar"):
            self.model = load_pretrained_torch_state_dict(self.model, OPT['pretrained'])
            log.info("Loaded `{}` pretrained model weights successfully.".format(OPT['pretrained']))
        elif OPT['pretrained'].endswith(".weights"):
            load_pretrained_darknet_state_dict(self.model, OPT['pretrained'])
            log.info("Loaded `{}` pretrained model weights successfully.".format(OPT['pretrained']))
        else:
            print("Pretrained model weights not found.")
        self.scaheduler = self.define_scheduler(self.optimizer, self.start_epoch, OPT['epochs'])
        self.tbw = SummaryWriter(os.path.join("logs", os.path.basename(OPT['net_cfg']).split('.')[0]))
        log.info('Start Tensorboard with "tensorboard --logdir=logs", view at http://localhost:6006/')
        iouv = torch.linspace(0.5, 0.95, 10).to(OPT['device'])  # iou vector for mAP@0.5:0.95
        self.iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
        self.niou = iouv.numel()

    def _load_options(self, path:str) -> None:
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """

        global HYP, OPT
        with open(path, 'r') as f:
            opt_full = yaml.load(f, Loader=yaml.CLoader)
        HYP = opt_full.get('hyper')
        OPT = opt_full.get('train')
        log.info(OPT)
        return

    def _build_dataset(self) -> tuple[Dataset, DataLoader, DataLoader, list, int]:
        # Load dataset
        dataset_dict = parse_dataset_config(OPT['data_cfg'])
        n_classes = 1 if OPT['single_cls'] else int(dataset_dict["classes"])
        names = dataset_dict["names"]
        HYP["cls"] *= n_classes / 80
        train_dataset = LoadImagesAndLabels(
            path=dataset_dict["train"],
            image_size=OPT['img_size_max'],
            batch_size=OPT['batch_size'],
            augment=OPT['augment'],
            hyper_parameters_dict=HYP,
            rect_label=OPT['rect_label'],
            cache_images=OPT['cache_imgs'],
            single_classes=OPT['single_cls'],
            gray=OPT['gray']
        )
        val_dataset = LoadImagesAndLabels(
            path=dataset_dict["valid"],
            image_size=OPT['val_img_size'],
            batch_size=OPT['batch_size'],
            augment=OPT['augment'],
            hyper_parameters_dict=HYP,
            rect_label=OPT['rect_label'],
            cache_images=OPT['cache_imgs'],
            single_classes=OPT['single_cls'],
            gray=OPT['gray']
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=OPT['batch_size'],
            shuffle=not OPT['rect_label'],
            num_workers=OPT['n_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=train_dataset.collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=OPT['batch_size'],
            shuffle=False,
            num_workers=OPT['n_workers'],
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            collate_fn=train_dataset.collate_fn
        )
        return train_dataset, train_dataloader, val_dataloader, names, n_classes

    def _build_model(self) -> tuple[nn.Module, nn.Module]:
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """

        # model = ROIEstNet(image_size=[OPT['img_size_max'], OPT['img_size_max']]).to(OPT['device'])
        model = Darknet(
            OPT['net_cfg'], 
            image_size=(OPT['img_size_max'], OPT['img_size_max']),
            gray=OPT['gray']
        )
        model = model.to(OPT['device'])
        model.num_classes = self.n_classes
        model.hyper_parameters_dict = HYP
        model.gr = 1.0
        model.class_weights = labels_to_class_weights(
            self.train_dataset.labels,
            1 if OPT['single_cls'] else self.n_classes
        )
        # Generate an exponential average model based on the generator to stabilize model training
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - OPT['ema_decay']) * averaged_model_parameter + OPT['ema_decay'] * model_parameter
        ema_model = AveragedModel(model, device=OPT['device'], avg_fn=ema_avg_fn)
        ema_model = ema_model.to(OPT['device'])

        return model, ema_model

    def add_batch_sample_to_tb(self, imgs:list, targets:list, paths:list, step:int) -> None:
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """

        tag = 'batch_sample'
        plotted_img = plot_images(images=imgs, targets=targets, paths=paths)
        if self.tbw:
            self.tbw.add_image(tag, plotted_img, dataformats='HWC', global_step=step)
        return

    def validate(self):
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """
        return Tester.test(
            model=self.model,
            test_dataloader=self.val_dataloader,
            names=self.names,
            conf_threshold=OPT['conf_threshold'],
            iou_threshold=OPT['iou_threshold'],
            augment=OPT['augment'],
            iouv=self.iouv,
            niou=self.niou,
            verbose=OPT['verbose'],
            device=OPT['device']
        )

    def train(
        self,
        epoch: int,
        batches: int,
        n_burn: int,
        print_freq: int = 1
    ) -> None:
        """training function

        Args:
            epoch (int): number of training epoch
            batches (int): number of total batches
            n_burn (int): number of burn-in batches
            print_frequency (int, optional): print frequency. Defaults to 1.

        Raises:
            None

        Returns:
            None
        """
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        giou_losses = AverageMeter("GIoULoss", ":6.6f")
        obj_losses = AverageMeter("ObjLoss", ":6.6f")
        cls_losses = AverageMeter("ClsLoss", ":6.6f")
        dep_losses = AverageMeter("DepLoss", ":6.6f")
        losses = AverageMeter("Loss", ":6.6f")
        progress = ProgressMeter(
            batches,
            [batch_time, data_time, giou_losses, obj_losses, cls_losses, dep_losses, losses],
            prefix=f"Epoch: [{epoch + 1}]"
        )
        self.model.train()
        end = time.time()
        accumulate = max(round(OPT['accumulate_batch_size'] / OPT['batch_size']), 1)
        for batch_i, (imgs, targets, paths, _, roi) in enumerate(self.train_dataloader):
            total_batch_i = batch_i + (batches * epoch) 
            imgs = imgs.to(OPT['device']).float() / 255.0  
            targets = targets.to(OPT['device'])
            data_time.update(time.time() - end)
            self.add_batch_sample_to_tb(imgs, targets, paths, 0) if total_batch_i==0 else None
            if total_batch_i <= n_burn:
                xi = [0, n_burn]
                self.model.gr = np.interp(total_batch_i, xi, [0.0, 1.0])
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    lr_decay = \
                        lambda lr: (((1 + math.cos(lr * math.pi / OPT['epochs'])) / 2) ** 1.0) * 0.95 + 0.05
                    x['lr'] = np.interp(
                        total_batch_i, xi, 
                        [0.1 if j == 2 else 0.0, x.get('initial_lr') * lr_decay(epoch)]
                    )
                    x['weight_decay'] = np.interp(
                        total_batch_i, xi, [0.0, OPT['optim_weight_decay'] if j == 1 else 0.0]
                    )
                    if 'momentum' in x:
                        x['momentum'] = np.interp(total_batch_i, xi, [0.9, OPT['optim_momentum']])
            self.model.zero_grad(set_to_none=True)
            # Mixed precision training
            with cuda.amp.autocast():
                p, p_roidepth = self.model(imgs, roi)
                loss, loss_item = compute_loss(p, p_roidepth, targets, self.model)
                loss *= OPT['batch_size'] / OPT['accumulate_batch_size']
            # Backpropagation
            self.scaler.scale(loss).backward()
            # update generator weights
            if total_batch_i % accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            self.ema_model.update_parameters(self.model)
            # update looger
            giou_losses.update(loss_item[0], imgs.size(0))
            obj_losses.update(loss_item[1], imgs.size(0))
            cls_losses.update(loss_item[2], imgs.size(0))
            dep_losses.update(loss_item[3], imgs.size(0))
            losses.update(loss_item[4], imgs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            # Record training log information
            if batch_i % print_freq == 0:
                # Writer Loss to file
                self.tbw.add_scalar("Train/GIoULoss", loss_item[0], total_batch_i)
                self.tbw.add_scalar("Train/ObjLoss", loss_item[1], total_batch_i)
                self.tbw.add_scalar("Train/ClsLoss", loss_item[2], total_batch_i)
                self.tbw.add_scalar("Train/DepLoss", loss_item[3], total_batch_i)
                self.tbw.add_scalar("Train/Loss", loss_item[4], total_batch_i)
                progress.display(batch_i)
        return

    @timer("training")
    def go(self) -> None:
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """

        best_map50 = 0.0
        n_batch = len(self.train_dataloader)
        log.info('Starting training for {} epochs...'.format(OPT['epochs']))

        for epoch in range(self.start_epoch, OPT['epochs']):
            self.train(
                epoch=epoch, 
                batches=n_batch,
                n_burn=max(3*n_batch, 500),
                print_freq=305
            )
            p, r, map50, f1, maps, dep_acc = self.validate()
            self.tbw.add_scalar("Val/Precision", p, epoch + 1)
            self.tbw.add_scalar("Val/Recall", r, epoch + 1)
            self.tbw.add_scalar("Val/mAP0.5", map50, epoch + 1)
            self.tbw.add_scalar("Val/F1", f1, epoch + 1)
            self.tbw.add_scalar("Val/Acc@dep", dep_acc, epoch + 1)
            self.scaheduler.step()
            # Automatically save model weights
            is_best = map50 > best_map50
            is_last = (epoch + 1) == OPT['epochs']
            best_map50 = max(map50, best_map50)
            save_torch_state_dict(
                {
                    "epoch": epoch + 1,
                    "best_map50": best_map50,
                    "state_dict": self.model.state_dict(),
                    "ema_state_dict": self.ema_model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                },
                f"epoch_{epoch + 1}.pth.tar",
                self.tbw.get_logdir(),
                self.tbw.get_logdir(),
                "best.pth.tar",
                "last.pth.tar",
                is_best,
                is_last
            )
        return

    def define_optimizer(self, model: nn.Module) -> optim.SGD:
        optim_group, weight_decay, biases = [], [], []  # optimizer parameter groups
        for k, v in dict(model.named_parameters()).items():
            if ".bias" in k:
                biases += [v]  # biases
            elif "Conv2d.weight" in k:
                weight_decay += [v]  # apply weight_decay
            else:
                optim_group += [v]  # all else
        optimizer = optim.SGD(
            optim_group,
            lr=OPT['optim_lr'],
            momentum=OPT['optim_momentum'],
            nesterov=True
        )
        optimizer.add_param_group({"params": weight_decay, "weight_decay": OPT['optim_weight_decay']})
        optimizer.add_param_group({"params": biases})
        del optim_group, weight_decay, biases

        return optimizer

    def define_scheduler(self, optimizer: optim.SGD, start_epoch: int, epochs: int) -> lr_scheduler.LambdaLR:
        """
        Define the learning rate scheduler
        Paper:
            https://arxiv.org/pdf/1812.01187.pdf
        Args:
            optimizer (optim.SGD): The optimizer to be used for training
            start_epoch (int): The epoch to start training from
            epochs (int): The total number of epochs to train for
        Returns:
            lr_scheduler.LambdaLR: The learning rate scheduler
            
        """
        lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler.last_epoch = start_epoch - 1

        return scheduler
    
class Tester(BaseTask):
    """
    pass

    Args:
       pass

    Attributes:
        pass

    """
    model: nn.Module = None
    dataloader: Any = None

    def __init__(self):
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """
        self._load_options('config.yaml')
        OPT['device'] = torch.device(OPT['device'])
        random.seed(OPT['seed'])
        np.random.seed(OPT['seed'])
        torch.manual_seed(OPT['seed'])
        torch.cuda.manual_seed_all(OPT['seed'])
        self.test_dataloader, self.names = self._build_dataset()
        self.model = self._build_model()
        iouv = torch.linspace(0.5, 0.95, 10).to(OPT['device'])  # iou vector for mAP@0.5:0.95
        self.iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
        self.niou = iouv.numel()

    def _load_options(self, path:str):
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """

        with open(path, 'r') as f:
            opt = yaml.load(f, Loader=yaml.CLoader).get('test')
        log.info(opt)

    def _build_dataset(self) -> tuple[nn.Module, int, list]:
        dataset_dict = parse_dataset_config(OPT['data_cfg'])
        names = load_classes(dataset_dict["names"])
        test_dataset = LoadImagesAndLabels(
            path=dataset_dict["valid"],
            image_size=OPT['img_size'],
            batch_size=OPT['batch_size'],
            augment=OPT['augment'],
            rect_label=OPT['rect_label'],
            cache_images=OPT['cache_imgs'],
            single_classes=OPT['single_cls'],
            pad=0.5,
            gray=OPT['gray']
        )
        # generate dataset iterator
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=OPT['batch_size'],
            shuffle=False,
            num_workers=OPT['num_workers'],
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            collate_fn=test_dataset.collate_fn
        )
        return test_dataloader, names

    def _build_model(self) -> nn.Module:
        n_classes = 1 if OPT['single_cls'] else int(parse_dataset_config(OPT['data_cfg'])["classes"])
        model = Darknet(OPT['net_cfg'], image_size=OPT['img_size']).to(OPT['device'])
        model.num_classes = n_classes
        if OPT['weights'].endswith(".pth.tar"):
            model = load_pretrained_torch_state_dict(model, OPT['weights'])
        elif OPT['weights'].endswith(".weights"):
            load_pretrained_darknet_state_dict(model, OPT['weights'])
        else:
            model = load_pretrained_torch_state_dict(model, OPT['weights'])
        log.info(f"Loaded `{OPT['weights']}` pretrained model weights successfully.")
        return model

    @classmethod
    def test(
        self,
        model: nn.Module,
        test_dataloader: DataLoader,
        names: list,
        conf_threshold: float,
        iou_threshold: float,
        augment: bool,
        iouv: torch.Tensor,
        niou: int,
        verbose: bool = False,
        device: torch.device = torch.device("cpu")
    ):
        seen = 0
        model.eval()
        # Format print information
        s = ("%20s" + "%10s" * 7) % ("Class", "Images", "Targets", "P", "R", "mAP@0.5", "F1", "Acc@dep")
        p, r, f1, mp, mr, map50, mf1 = 0., 0., 0., 0., 0., 0., 0.
        jdict, stats, ap, ap_class, dep_errs = [], [], [], [],[]
        for _, (imgs, targets, _, _, roi) in enumerate(tqdm.tqdm(test_dataloader, desc=s)):
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            _, _, height, width = imgs.shape  # batch size, channels, height, width
            whwh = torch.Tensor([width, height, width, height]).to(device)
            with torch.no_grad():
                output, _, depth_output = model(imgs, roi)  # inference and training outputs
                output = non_max_suppression(output, conf_threshold, iou_threshold)
            # Statistics per image
            for si, pred in enumerate(zip(output, depth_output)):
                pred_obj, pred_dep = pred
                labels = targets[targets[:, 0] == si, 1:]
                n_labels = len(labels)
                target_classes = labels[:, 0].tolist() if n_labels else []  # target class
                seen += 1
                if pred_obj is None:
                    if n_labels:
                        stats.append(
                            (
                                torch.zeros(0, niou, dtype=torch.bool),
                                torch.Tensor(),
                                torch.Tensor(),
                                target_classes
                            )
                        )
                    continue
                # Clip boxes to image bounds
                clip_coords(pred_obj, (height, width))
                # Assign all predictions as incorrect
                correct = torch.zeros(pred_obj.shape[0], niou, dtype=torch.bool, device=device)
                if n_labels:
                    detected = []  # target indices
                    target_classes_tensor = labels[:, 0]
                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5]) * whwh
                    # Per target class
                    for cls in torch.unique(target_classes_tensor):
                        ti = (cls == target_classes_tensor).nonzero().view(-1)  # target indices
                        pi = (cls == pred_obj[:, 5]).nonzero().view(-1)  # prediction indices
                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = boxes.box_iou(pred_obj[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            # Append detections
                            for j in (ious > iouv[0]).nonzero():
                                d = ti[i[j]]  # detected target
                                if d not in detected:
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv
                                    if len(detected) == n_labels:  # all targets already located in image
                                        break
                dep_err = abs(labels[:, 5] - pred_dep).cpu()
                dep_errs.append(dep_err)
                # Append statistics (correct, conf, pcls, target_classes)
                stats.append((correct.cpu(), pred_obj[:, 4].cpu(), pred_obj[:, 5].cpu(), target_classes))
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats):
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            if niou > 1:
                p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
            mp, mr, map50, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=model.num_classes)  # number of targets per class
        else:
            nt = torch.zeros(1)
        # dep_acc = np.mean([e[0] for e in dep_errs]) if len(dep_errs)!=0 else 0
        dep_acc = np.mean([e[0] for e in dep_errs])
        # Print results
        pf = "%20s" + "%10.3g" * 7  # print format
        print(pf % ("all", seen, nt.sum(), mp, mr, map50, mf1, dep_acc))
        # Print results per class
        if verbose and model.num_classes > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
        # Return results
        maps = np.zeros(model.num_classes) + map50
        for ap_index, c in enumerate(ap_class):
            maps[c] = ap[ap_index]
        return mp, mr, map50, mf1, maps, dep_acc

    def go(self):
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """
        pass


class Detector(BaseTask):
    """
    pass

    Args:
       pass

    Attributes:
        pass

    """

    def run(self):
        """pass

        Args:
            pass

        Raises:
            pass

        Returns:
            pass
        """

if __name__=="__main__":
    # Train
    task = Trainer()
    task.go()
    exit(0)