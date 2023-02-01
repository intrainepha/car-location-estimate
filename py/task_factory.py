import os
import random
import math
import yaml
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Callable, Any
from utils.torch_utils import select_device, ModelEMA, time_synchronized
from models import Darknet, YOLOLayer, F, parse_data_cfg, load_darknet_weights
from utils.datasets import ImagesAndLabelsLoader
from utils.utils import init_seeds, labels_to_class_weights, load_classes, \
    labels_to_image_weights, compute_loss, plot_images, fitness, check_file, \
    non_max_suppression, clip_coords, scale_coords, xyxy_to_xywh, xywh_to_xyxy, \
    box_iou, output_to_target, ap_per_class
from utils.wrapper import timer
from indicators import collect_depth, cal_depth_indicators
from typing import Union, Dict, Any
import absl.logging as log
log.set_verbosity(log.INFO)

Y_RANGE = 80

def get(task:str='train') -> Callable:
    """TODO

    Args:
        TODO

    Raises:
        TODO

    Returns:
        TODO
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
    TODO

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
        save_json (bool):     save a cocoapi-compatible JSON results file
        test_mode (str):      'test', 'study', 'benchmark'
        augment (bool):       augmented inference
        names (str):          *.names path'
        source (str):         input file/folder, 0 for webcam
        output (str):         output folder
        fourcc (str):         output video codec (verify ffmpeg support)
        half (bool):          half precision FP16 inference
        view_img (bool):      display results
        save_txt (bool):      save results to *.txt'
        classes (int):        filter by class
        agnostic_nms(bool):   class-agnostic NMS
    """

    @abstractmethod
    def load_options(self) -> Dict[str, Any]:
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        pass

    @abstractmethod
    def run(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        pass

class Trainer(BaseTask):
    """
    TODO

    Args:
       TODO

    Attributes:
        TODO

    """

    dataset: ImagesAndLabelsLoader = None
    trainloader: Any = None
    model: Darknet = None
    optimizer: Any = None
    scheduler: Any = None
    ema: ModelEMA = None
    start_epoch: int = 0

    def __call__(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        self.load_options('config.yaml')
        check_file(self.cfg)
        check_file(self.data)
        self.data_dict = parse_data_cfg(self.data)
        self.device = select_device(self.device, batch_size=self.batch_size)
        init_seeds()
        log.info('Image sizes {} - {}'.format(self.imsz[0], self.imsz[1]))
        from torch.utils.tensorboard import SummaryWriter
        log.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        self.tbw = SummaryWriter()
        self.lf = lambda x: (((1 + math.cos(x * math.pi / self.epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
        self.load_hpy('config.yaml')
        self.load_dataset()
        self.init_model()
        self.ema = ModelEMA(self.model)
        self.validator = Tester(model=self.model, dataloader=self.valloader)
        self.run() 

    def load_options(self, path:str):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        with open(path, 'r') as f:
            opt = yaml.load(f, Loader=yaml.CLoader).get('train')
        self.cfg = os.path.abspath(opt.get('cfg'))
        self.data = os.path.abspath(opt.get('data'))
        self.weights = os.path.abspath(opt.get('weights'))
        self.imsz = opt.get('img_size')
        self.device = str(opt.get('device'))
        self.epochs = opt.get('epochs')
        self.batch_size = opt.get('batch_size')
        # self.grid_size = opt.get('grid_size')
        self.multi_scale = opt.get('multi_scale')
        self.rect = opt.get('rect')
        self.cache_imgs = opt.get('cache_imgs')
        self.single_cls = opt.get('single_cls')
        self.freeze_layers = opt.get('freeze_layers')
        log.info(opt)
        self.accumulate = max(round(64 / self.batch_size), 1) 

    def load_hpy(self, path:str):
        global HYP
        with open(path, 'r') as f:
            HYP = yaml.load(f, Loader=yaml.CLoader).get('hyper')

    def init_model(self) -> Any:
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        self.model = Darknet(self.cfg).to(self.device)
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(self.model.named_parameters()).items():
            if '.bias' in k:
                pg2 += [v]  # biases
            elif 'Conv2d.weight' in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else
        self.optimizer = optim.Adam(pg0, lr=HYP.get('lr0'))
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': HYP.get('weight_decay')})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        log.info('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2
        if self.weights.endswith('.pt'):
            ckpt = torch.load(self.weights, map_location=self.device)
            try:
                ckpt['model'] = {
                    k: v for k, v in ckpt['model'].items()
                        if self.model.state_dict()[k].numel() == v.numel()
                }
                self.model.load_state_dict(ckpt['model'], strict=False)
            except KeyError as e:
                s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                    "See https://github.com/ultralytics/yolov3/issues/657" % (self.weights, self.cfg, self.weights)
                raise KeyError(s) from e
            # load optimizer
            if ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']
            # epochs
            self.start_epoch = ckpt['epoch'] + 1
            if self.epochs < self.start_epoch:
                log.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                    (self.weights, ckpt['epoch'], self.epochs))
                self.epochs += ckpt['epoch']
            del ckpt
        elif len(self.weights) > 0: 
            load_darknet_weights(self.model, self.weights)
        if self.freeze_layers:
            output_layer_indices = [
                idx - 1 for idx, module in enumerate(self.model.module_list) if isinstance(module, YOLOLayer)
            ]
            freeze_layer_indices = [
                x for x in range(len(self.model.module_list)) 
                if (x not in output_layer_indices) and (x - 1 not in output_layer_indices)
            ]
            for idx in freeze_layer_indices:
                for parameter in self.model.module_list[idx].parameters():
                    parameter.requires_grad_(False)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.scheduler.last_epoch = self.start_epoch - 1  # see link below
        if self.device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
            dist.init_process_group(
                backend='nccl',                      # 'distributed backend'
                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                world_size=1,                        # number of nodes for distributed training
                rank=0                               # distributed training node rank
            )
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
            self.model.yolo_layers = self.model.module.yolo_layers  # move yolo layer indices to top level
            # Model parameters
        self.model.nc = 1 if self.single_cls else int(self.data_dict.get('classes'))   # attach number of classes to model
        HYP['cls'] *= self.model.nc / 80  # update coco-tuned HYPER_PARAMS['cls'] to current dataset
        self.model.hyp = HYP # attach hyperparameters to model
        self.model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.model.nc).to(self.device)  # attach class weights


    def load_dataset(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        self.dataset = ImagesAndLabelsLoader(
            self.data_dict.get('train'),
            self.imsz[1],
            self.batch_size,
            augment=False,
            hyp=HYP, 
            rect=self.rect,  # rectangular training
            cache_images=self.cache_imgs,
            single_cls=self.single_cls
        )
        # Dataloader
        self.batch_size = min(self.batch_size, len(self.dataset))
        n_workers = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])
        log.info('Using {} dataloader workers'.format(n_workers))
        self.trainloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=n_workers,
            shuffle=not self.rect,
            pin_memory=True,
            collate_fn=self.dataset.collate_fn
        )

        self.valloader = torch.utils.data.DataLoader(
            ImagesAndLabelsLoader(
                path=self.data_dict.get('valid'),
                img_size=self.imsz[1],
                batch_size=16,
                augment=False,
                hyp=HYP,
                # rect=True,
                cache_images=self.cache_imgs,
                single_cls=self.single_cls
            ),
            batch_size=16,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=self.dataset.collate_fn
        )

    def add_tb_info(self, imgs:list, targets:list, paths:list, step:int):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        tag = 'batch_sample'
        im = plot_images(images=imgs, targets=targets, paths=paths, fname=tag)
        if self.tbw:
            self.tbw.add_image(tag, im, dataformats='HWC', global_step=step)
            # self.tbw.add_graph(self.model, imgs) 

    def validate(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        # results, maps = test.test(
        #     cfg,
        #     data,
        #     batch_size=batch_size,
        #     imgsz=imgsz_test,
        #     model=ema.ema,
        #     save_json=final_epoch and is_coco,
        #     single_cls=opt.single_cls,
        #     dataloader=testloader,
        #     multi_label=ni > n_burn
        # )
        self.validator()



    @timer("training")
    def run(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        log.info('Starting training for {} epochs...'.format(self.epochs))
        n_batches = len(self.trainloader)
        n_burn = max(3 * n_batches, 500)
        results, maps = (0, 0, 0, 0, 0, 0, 0), np.zeros(self.model.nc) 
        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            if self.dataset.image_weights:
                w = self.model.class_weights.cpu().numpy() * (1 - maps) ** 2 
                image_weights = labels_to_image_weights(self.dataset.labels, nc=self.nc, class_weights=w)
                self.dataset.indices = random.choices(range(self.dataset.n), weights=image_weights, k=self.dataset.n) 
            mean_loss = torch.zeros(5).to(self.device) 
            log.info(
                ('\n' + '%10s' * 9) % 
                ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'dep', 'total', 'targets', 'img_size')
            ) 
            best_fitness = 0.0
            pbar = tqdm(enumerate(self.trainloader), total=n_batches) 
            for i, (imgs, targets, paths, _, roi_info) in pbar:
                ni = i + n_batches * epoch 
                imgs = imgs.to(self.device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
                targets = targets.to(self.device)
                if ni <= n_burn:
                    xi = [0, n_burn]
                    self.model.gr = np.interp(ni, xi, [0.0, 1.0])
                    self.accumulate = max(1, np.interp(ni, xi, [1, 64 / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        x['weight_decay'] = np.interp(ni, xi, [0.0, HYP.get('weight_decay') if j == 1 else 0.0])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [0.9, HYP.get('momentum')])
                pred, pred_depth = self.model(imgs, roi=roi_info)
                loss, loss_items = compute_loss(pred, pred_depth, targets, self.model)
                if not torch.isfinite(loss):
                    log.info('WARNING: non-finite loss, ending training ', loss_items)
                    return results
                loss *= self.batch_size / 64 
                loss.backward()
                if ni % self.accumulate == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.ema.update(self.model)
                mean_loss = (mean_loss * i + loss_items) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  
                s = ('%10s' * 2 + '%10.3g' * 7) % ('%g/%g' % (epoch, self.epochs - 1), mem, *mean_loss, len(targets), self.imsz[1]) 
                pbar.set_description(s)
                self.add_tb_info(imgs, targets, paths, epoch) if ni==0 else None
            self.scheduler.step()
            self.ema.update_attr(self.model)
            final_epoch = epoch + 1 == self.epochs
            self.validator()
            tags = [
                'train/giou_loss', 'train/obj_loss','train/cls_loss', 'train/dep_loss', 'train/total_loss', 
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1', 
                'val/giou_loss', 'val/obj_loss', 'val/cls_loss'
            ]
            for tag, x in zip(tags, list(mean_loss) + list(results)):
                self.tbw.add_scalar(tag, x, epoch)
            fi = fitness(np.array(results).reshape(1, -1))
            best_fitness = fi if fi > best_fitness else best_fitness
            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'training_results': mean_loss,
                'model': self.ema.ema.module.state_dict() if hasattr(self.model, 'module') else self.ema.ema.state_dict(),
                'optimizer': None if final_epoch else self.optimizer.state_dict()
            }
            tb_dir = os.path.abspath(self.tbw.get_logdir())
            torch.save(ckpt, os.path.join(tb_dir, 'last.pt'))
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, os.path.join(tb_dir, 'best.pt'))
                log.info('Save as best')
            del ckpt
        dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
        torch.cuda.empty_cache()
        return results

    # def extend_img_size(self):
    #     """TODO

    #     Args:
    #         TODO

    #     Raises:
    #         TODO

    #     Returns:
    #         TODO
    #     """

    #     # Extend to 3 sizes (min, max, test)
    #     self.imsz.extend([self.imsz[-1]] * (3 - len(self.imsz)))  
    #     assert math.fmod(self.imsz[0], self.grid_size) ==  0, \
    #         'image size {} must be a {}-multiple'.format(self.imsz, self.grid_size)

@dataclass
class Tester(BaseTask):
    """
    TODO

    Args:
       TODO

    Attributes:
        TODO

    """
    model: Darknet = None
    dataloader: Any = None

    def __call__(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """
        self.load_options('config.yaml')
        if self.model is None:
            is_training = False
            self.device = select_device(self.device, batch_size=self.batch_size)
            # for f in glob.glob('test_batch*.jpg'):
            #     os.remove(f)
            model = Darknet(self.cfg, self.imsz)
            if self.weights.endswith('.pt'):  
                model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
            else:  
                load_darknet_weights(model, self.weights)
            # Fuse
            model.fuse()
            model.to(self.device)
            if self.device.type != 'cpu' and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        else:  # called by train.py
            is_training = True
            self.device = next(self.model.parameters()).device  
            verbose = False
        # Configure run
        data = parse_data_cfg(self.data)
        nc = 1 if self.single_cls else int(data['classes'])  
        path = data['valid']  
        names = load_classes(data['names'])  
        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
        iouv = torch.Tensor([0.35]).to(self.device) # adaption
        niou = iouv.numel()
        # Dataloader
        if self.dataloader is None:
            dataset = ImagesAndLabelsLoader(path, self.imsz, batch_size, single_cls=self.single_cls, pad=0.5)
            batch_size = min(batch_size, len(dataset))
            self.dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                pin_memory=True,
                collate_fn=dataset.collate_fn
            )
        seen = 0
        self.model.eval()
        _ = self.model(torch.zeros((1, 3, self.imsz, self.imsz), device=self.device)) if self.device.type != 'cpu' else None  # run once
        s = ('%20s' + '%10s' * 9) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1', 'derror', 'dacc', 'dstd')
        p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        loss = torch.zeros(3, device=self.device)
        jdict, stats, ap, ap_class = [], [], [], []
        d_error, d_acc = [], [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        # depth_stats = [[],[],[],[],[]] # adaption, for saving depth error stats with 5 different ranges(in merters):[0,10]/[10,30]/[30/50]/[50,80]/[80,150].
        de_acc = [[],[],[],[],[]]
        for batch_i, (imgs, targets, paths, shapes, roi_info) in enumerate(tqdm(self.dataloader, desc=s)): # adaption, original:"for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):"
            imgs = imgs.to(self.device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(self.device)
            nb, _, height, width = imgs.shape  # batch size, channels, height, width
            whwh = torch.Tensor([width, height, width, height]).to(self.device)
            # Disable gradients
            with torch.no_grad():
                # Run model
                t = time_synchronized()
                inf_out, train_out, roidepth_out = self.model(imgs, roi=roi_info, augment=self.augment)  # inference and training outputs. adaption, original:"inf_out, train_out, roidepth_out = model(imgs, augment=augment)"
                t0 += time_synchronized() - t
                # Compute loss
                if is_training:  # if model has loss hyperparameters
                    loss += compute_loss(train_out, roidepth_out, targets, self.model)[1][:3]  # GIoU, obj, cls
                # Run NMS
                t = time_synchronized()
                output = non_max_suppression(
                    inf_out,
                    conf_thres=self.conf_thres,
                    iou_thres=self.iou_thres,
                    multi_label=self.multi_label
                )
                t1 += time_synchronized() - t
            # Statistics per image
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1
                if pred is None:
                    if nl:
                        stats.append(
                            (torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls)
                        )
                    continue
                # Append to text file
                # with open('test.txt', 'a') as file:
                #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))
                # Append to pycocotools JSON dictionary
                if self.save_json:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(Path(paths[si]).stem.split('_')[-1])
                    box = pred[:, :4].clone()  # xyxy
                    scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                    box = xyxy_to_xywh(box)  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append(
                            {
                                'image_id': image_id,
                                # 'category_id': coco91class[int(p[5])],
                                'bbox': [round(x, 3) for x in b],
                                'score': round(p[4], 5)
                            }
                        )
                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=self.device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]
                    # target boxes
                    tbox = xywh_to_xyxy(labels[:, 1:5]) * whwh
                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices
                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            # Append detections
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d not in detected:
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break
                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
                # Depth stats for eval, adaption.
                tdepth = labels[:,5]
                pdepth, tdepth = np.squeeze(
                    roidepth_out.cpu().numpy()
                )*Y_RANGE, tdepth.cpu().numpy()*Y_RANGE
                # de = abs(tdepth-pdepth)
                # if 1<tdepth<=10: # 0~10m
                #     depth_stats[0].append(de)
                #     de_acc[0].append(1-de/tdepth)
                # elif 10<tdepth<=30: # 10~30m
                #     depth_stats[1].append(de)
                #     de_acc[1].append(1-de/tdepth)
                # elif 30<tdepth<=50: # 30~50m
                #     depth_stats[2].append(de)
                #     de_acc[2].append(1-de/tdepth)
                # elif 50<tdepth<=80: # 50~80m
                #     depth_stats[3].append(de)
                #     de_acc[3].append(1-de/tdepth)
                # else:# 80~150m
                #     depth_stats[4].append(de)
                #     de_acc[4].append(1-de/tdepth)

                collect_depth(d_error, d_acc, tdepth, pdepth)
            # Plot images
            if batch_i < 1:
                f = 'test_batch%g_gt.jpg' % batch_i  # filename
                plot_images(imgs, targets, paths=paths, names=names, fname=f)  # ground truth
                f = 'test_batch%g_pred.jpg' % batch_i
                plot_images(
                    imgs, output_to_target(output, width, height), paths=paths, names=names, fname=f
                )  # predictions
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats):
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            if niou > 1:
                p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Eval depth, adaption 
        # de_mean = [np.mean(ds) for ds in depth_stats] # get mean error in different range
        # de_acc = [np.mean(da) for da in de_acc] # get accuracy error in different range
        # de_std = [np.std(me) for me in de_mean] # get standard deviation of depth error in different range
        # len_stats = [len(depth_stats[0]), len(depth_stats[1]), len(depth_stats[2]), len(depth_stats[3]), len(depth_stats[4])]
        # total_mean = sum([x*y if y>0 else 0 for x,y in zip(de_mean, len_stats)])/sum(len_stats)
        # total_acc = sum([x*y if y>0 else 0 for x,y in zip(de_acc, len_stats)])/sum(len_stats)
        # total_std = sum([x*y if y>0 else 0 for x,y in zip(de_std, len_stats)])/sum(len_stats)
        # depth_stats = [str(len_stats)+'\n', str(de_mean)+'\n', 
        #                 str(de_acc)+'\n', str(de_std)+'\n']
        # with open('depth-test-stats.txt', 'w', encoding='utf-8') as f:
        #     f.writelines(depth_stats)

        # cal_depth_indicators(d_error, d_acc)
        # Print results
        pf = '%20s' + '%10.3g' * 6  
        log.info(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))
        # Print results per class
        if verbose and nc > 1 and len(stats):
            pfc = '%20s' + '%10.3g' * 6  # print format
            for i, c in enumerate(ap_class):
                log.info(pfc % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
        # Print speeds
        if verbose or self.save_json:
            t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (self.imsz, self.imsz, batch_size)  # tuple
            log.info('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

        # # Save JSON
        # if save_json and map and len(jdict):
        #     log.info('\nCOCO mAP with pycocotools...')
        #     imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        #     with open('results.json', 'w') as file:
        #         json.dump(jdict, file)

        #     try:
        #         from pycocotools.coco import COCO
        #         from pycocotools.cocoeval import COCOeval

        #         # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        #         cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
        #         cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

        #         cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        #         cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        #         cocoEval.evaluate()
        #         cocoEval.accumulate()
        #         cocoEval.summarize()
        #         # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        #     except:
        #         log.info('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
        #               'See https://github.com/cocodataset/cocoapi/issues/356')

        # Return results
        maps = np.zeros(nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        return (mp, mr, map, mf1, *(loss.cpu() / len(self.dataloader)).tolist()), maps
    
    def load_options(self, path:str):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        with open(path, 'r') as f:
            opt = yaml.load(f, Loader=yaml.CLoader).get('test')
        self.cfg = os.path.abspath(opt.get('cfg'))
        self.data = os.path.abspath(opt.get('data'))
        self.weights = os.path.abspath(opt.get('weights'))
        self.verbose = opt.get('verbose')
        self.imsz = opt.get('img_size')
        self.device = str(opt.get('device'))
        self.batch_size = opt.get('batch_size')
        self.conf_thres = opt.get('conf_thres')
        self.iou_thres = opt.get('iou_thres')
        self.save_json = opt.get('save_json')
        self.single_cls = opt.get('single_cls')
        self.augment = opt.get('augment')
        self.multi_label = opt.get('multi_label')
        log.info(opt)

    def run(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """
        pass


class Detector(BaseTask):
    """
    TODO

    Args:
       TODO

    Attributes:
        TODO

    """

    def run(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

if __name__=="__main__":
    # Train
    T = get(task="train")
    task = T()
    task()