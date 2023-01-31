import os
import random
import math
import yaml
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from glob import glob
from time import time
from tqdm import tqdm
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Callable, Any
from utils.torch_utils import select_device, ModelEMA
from models import Darknet, YOLOLayer, F, parse_data_cfg, \
    attempt_download, load_darknet_weights
from utils.datasets import ImagesAndLabelsLoader
from utils.utils import init_seeds, labels_to_class_weights, labels_to_image_weights, \
    compute_loss, plot_images, plot_results, fitness, check_file, strip_optimizer, \
    print_mutation, plot_evolution_results 
from typing import Union, Dict, Any
import absl.logging as log
log.set_verbosity(log.INFO)

WEIGHTS_DIR  = 'weights'
LAST         = os.path.join(WEIGHTS_DIR, 'last.pt')
BEST         = os.path.join(WEIGHTS_DIR, 'best.pt')
RES_FILE     = 'results.txt'
HYPER_PARAMS = dict(
    giou         = 3.54,     # giou loss gain
    cls          = 37.4,     # cls loss gain
    cls_pw       = 1.0,      # cls BCELoss positive_weight
    obj          = 64.3,     # obj loss gain (*=imsz/320 if imsz != 320)
    obj_pw       = 1.0,      # obj BCELoss positive_weight
    iou_t        = 0.20,     # iou training threshold
    lr0          = 0.001,    # initial learning rate (SGD=5E-3, Adam=5E-4)
    lrf          = 0.0005,   # final learning rate (with cos scheduler)
    momentum     = 0.937,    # SGD momentum
    weight_decay = 0.01,     # optimizer weight decay
    fl_gamma     = 0.0,      # focal loss gamma (efficientDet default is gamma=1.5)
    hsv_h        = 0.0138,   # image HSV-Hue augmentation (fraction)
    hsv_s        = 0.678,    # image HSV-Saturation augmentation (fraction)
    hsv_v        = 0.36,     # image HSV-Value augmentation (fraction)
    degrees      = 1.98 * 0, # image rotation (+/- deg)
    translate    = 0.05 * 0, # image translation (+/- fraction)
    scale        = 0.05 * 0, # image scale (+/- gain)
    shear        = 0.641 * 0 # image shear (+/- deg)
)  

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

@dataclass
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
    def get_opts(self) -> Dict[str, Any]:
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

    def __call__(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        self.get_opts('config.yaml')
        check_file(self.cfg)
        check_file(self.data)
        self.data_dict = parse_data_cfg(self.data)
        self.device = select_device(self.device, batch_size=self.batch_size)
        init_seeds()
        from torch.utils.tensorboard import SummaryWriter
        log.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        # self.tbw = SummaryWriter(comment=self.name)
        self.tbw = SummaryWriter()
        self.lf = lambda x: (((1 + math.cos(x * math.pi / self.epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
        self.run() 

    def get_opts(self, path:str) -> Dict[str, Any]:
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
        self.grid_size = opt.get('grid_size')
        self.multi_scale = opt.get('multi_scale')
        self.rect = opt.get('rect')
        self.cache_imgs = opt.get('cache_imgs')
        # self.name = opt.get('name')
        self.adam = opt.get('adam')
        self.single_cls = opt.get('single_cls')
        self.freeze_layers = opt.get('freeze_layers')
        log.info(opt)

        self.accumulate = max(round(64 / self.batch_size), 1) 
        
        return opt

    def init_model(self, dataset) -> Any:
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        model = Darknet(self.cfg).to(self.device)
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(model.named_parameters()).items():
            if '.bias' in k:
                pg2 += [v]  # biases
            elif 'Conv2d.weight' in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else
        optimizer = optim.Adam(pg0, lr=HYPER_PARAMS['lr0'])
        optimizer.add_param_group({'params': pg1, 'weight_decay': HYPER_PARAMS['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        log.info('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2
        start_epoch = 0
        if self.weights.endswith('.pt'):
            ckpt = torch.load(self.weights, map_location=self.device)
            try:
                ckpt['model'] = {
                    k: v for k, v in ckpt['model'].items()
                    if model.state_dict()[k].numel() == v.numel()
                }
                model.load_state_dict(ckpt['model'], strict=False)
            except KeyError as e:
                s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                    "See https://github.com/ultralytics/yolov3/issues/657" % (self.weights, self.cfg, self.weights)
                raise KeyError(s) from e
            # load optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']
            # load results
            if ckpt.get('training_results') is not None:
                with open(RES_FILE, 'w') as file:
                    file.write(ckpt['training_results'])
            # epochs
            start_epoch = ckpt['epoch'] + 1
            if self.epochs < start_epoch:
                log.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                    (self.weights, ckpt['epoch'], self.epochs))
                self.epochs += ckpt['epoch']
            del ckpt
        elif len(self.weights) > 0: 
            load_darknet_weights(model, self.weights)
        if self.freeze_layers:
            output_layer_indices = [
                idx - 1 for idx, module in enumerate(model.module_list) if isinstance(module, YOLOLayer)
            ]
            freeze_layer_indices = [
                x for x in range(len(model.module_list)) 
                if (x not in output_layer_indices) and (x - 1 not in output_layer_indices)
            ]
            for idx in freeze_layer_indices:
                for parameter in model.module_list[idx].parameters():
                    parameter.requires_grad_(False)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf)
        scheduler.last_epoch = start_epoch - 1  # see link below
        if self.device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
            dist.init_process_group(
                backend='nccl',                      # 'distributed backend'
                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                world_size=1,                        # number of nodes for distributed training
                rank=0                               # distributed training node rank
            )
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level
            # Model parameters
        nc = 1 if self.single_cls else int(self.data_dict.get('classes'))  
        HYPER_PARAMS['cls'] *= nc / 80  # update coco-tuned HYPER_PARAMS['cls'] to current dataset
        model.nc = nc   # attach number of classes to model
        model.hyp = HYPER_PARAMS # attach hyperparameters to model
        model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(self.device)  # attach class weights
        return model, optimizer, scheduler, nc, start_epoch

    def load_dataset(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        dataset = ImagesAndLabelsLoader(
            self.data_dict.get('train'),
            self.imsz[1],
            self.batch_size,
            augment=False,
            hyp=HYPER_PARAMS,              # augmentation hyperparameters
            rect=self.rect,                 # rectangular training
            cache_images=self.cache_imgs,
            single_cls=self.single_cls
        )
        # Dataloader
        self.batch_size = min(self.batch_size, len(dataset))
        nw = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=nw,
            shuffle=not self.rect,
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )
        return dataset, dataloader, nw

    def plot_batch_sample(self, imgs:list, targets:list, paths:list, step:int):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        tag = 'batch_sample.jpg'
        im = plot_images(images=imgs, targets=targets, paths=paths, fname=tag)
        if self.tbw:
            self.tbw.add_image(tag, im, dataformats='HWC', global_step=step)

    def run(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        grid_min, grid_max = self.extend_img_size()
        imsz = self.imsz[1]
        dataset, dataloader, nw = self.load_dataset()
        model, optimizer, scheduler, nc, start_epoch = self.init_model(dataset)
        log.info('Image sizes {} - {} train, {} test'.format(self.imsz[0], self.imsz[1], self.imsz[2]))
        log.info('Using {} dataloader workers'.format(nw))
        log.info('Starting training for {} epochs...'.format(self.epochs))
        ema = ModelEMA(model)
        n_bs = len(dataloader)
        n_burn = max(3 * n_bs, 500)
        results, maps = (0, 0, 0, 0, 0, 0, 0), np.zeros(nc) 
        t0 = time()
        for epoch in range(start_epoch, self.epochs):
            model.train()
            if dataset.image_weights:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2 
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n) 
            mean_loss = torch.zeros(5).to(self.device) 
            log.info(
                ('\n' + '%10s' * 9) % 
                ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'depth', 'targets', 'img_size')
            ) 
            best_fitness = 0.0
            pbar = tqdm(enumerate(dataloader), total=n_bs) 
            for i, (imgs, targets, paths, _, roi_info) in pbar:
                ni = i + n_bs * epoch 
                imgs = imgs.to(self.device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
                targets = targets.to(self.device)
                # Burn-in
                if ni <= n_burn:
                    xi = [0, n_burn]
                    model.gr = np.interp(ni, xi, [0.0, 1.0])
                    self.accumulate = max(1, np.interp(ni, xi, [1, 64 / self.batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        x['weight_decay'] = np.interp(ni, xi, [0.0, HYPER_PARAMS['weight_decay'] if j == 1 else 0.0])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [0.9, HYPER_PARAMS['momentum']])
                # Multi-Scale
                if self.multi_scale:
                    if ni / self.accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                        imsz = random.randrange(grid_min, grid_max + 1) * self.grid_size
                    sf = imsz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        # new shape (stretched to 32-multiple)
                        ns = [math.ceil(x * sf / self.grid_size) * self.grid_size for x in imgs.shape[2:]]  
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                pred, pred_depth = model(imgs, roi=roi_info)
                loss, loss_items = compute_loss(pred, pred_depth, targets, model)
                if not torch.isfinite(loss):
                    log.info('WARNING: non-finite loss, ending training ', loss_items)
                    return results
                loss *= self.batch_size / 64 
                loss.backward()
                if ni % self.accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    ema.update(model)
                mean_loss = (mean_loss * i + loss_items) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  
                s = ('%10s' * 2 + '%10.3g' * 7) % ('%g/%g' % (epoch, self.epochs - 1), mem, *mean_loss, len(targets), imsz) 
                pbar.set_description(s)
                self.plot_batch_sample(imgs, targets, paths, epoch) if ni==0 else None
            scheduler.step()
            ema.update_attr(model)
            final_epoch = epoch + 1 == self.epochs
            if self.tbw:
                tags = [
                    'train/giou_loss', 'train/obj_loss',
                    'train/cls_loss', 'metrics/precision',
                    'metrics/recall', 'metrics/mAP_0.5',
                    'metrics/F1', 'val/giou_loss',
                    'val/obj_loss', 'val/cls_loss'
                ]
                for x, tag in zip(list(mean_loss[:-1]) + list(results), tags):
                    self.tbw.add_scalar(tag, x, epoch)
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1)) # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi
            # Save model
            # with open(RES_FILE, 'r') as f:  # create checkpoint
            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                # 'training_results': f.read(),
                'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                'optimizer': None if final_epoch else optimizer.state_dict()
            }
            torch.save(ckpt, LAST)
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, BEST)
                log.info('Save as best')
            del ckpt
        log.info(
            '%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time() - t0) / 3600)
        )
        dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
        torch.cuda.empty_cache()
        return results

    def extend_img_size(self) -> Union[float, float]:
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        # Extend to 3 sizes (min, max, test)
        self.imsz.extend([self.imsz[-1]] * (3 - len(self.imsz)))  
        assert math.fmod(self.imsz[0], self.grid_size) ==  0, \
            'image size {} must be a {}-multiple'.format(self.imsz, self.grid_size)
        self.multi_scale |= self.imsz[0] != self.imsz[1]  
        grid_min, grid_max = 0, 0
        if self.multi_scale:
            if self.imsz[0] == self.imsz[1]:
                self.imsz[0] //= 1.5
                self.imsz[1] //= 0.667
            grid_min, grid_max = self.imsz[0] // self.grid_size, self.imsz[1] // self.grid_size
            self.imsz[0], self.imsz[1] = \
                int(grid_min * self.grid_size), int(grid_max * self.grid_size)
        return grid_min, grid_max

class Tester(BaseTask):
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
    T = get(task="train")
    task = T()
    task()