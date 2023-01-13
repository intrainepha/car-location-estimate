import os
import numpy as np
import random
from time import time
from math import fmod
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Callable, Any
from utils.torch_utils import select_device
from utils.utils import init_seeds, labels_to_class_weights, \
    labels_to_image_weights, compute_loss, plot_images, plot_results, \
    fitness, check_git_status, check_file, strip_optimizer, \
    print_mutation, plot_evolution_results 
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
    obj          = 64.3,     # obj loss gain (*=img_size/320 if img_size != 320)
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

def TaskFactory(task:str='train') -> Callable:
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
        img_size (int):       [min, max, test]
        weights (str):        initial weights path
        device (str):         device id (i.e. 0 or 0,1 or cpu)
        epochs (int): 
        batch-size (int):
        data (str):           *.data path
        multi_scale (bool):   adjust (67%% - 150%%) img_size every 10 batches
        rect (bool):          rectangular training
        resume (bool):        resume training from last.pt
        nosave (bool):        only save final checkpoint
        notest (bool):        only test final epoch
        evolve (bool):        evolve hyperparameters
        bucket (str):         gsutil bucket
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

    cfg: str            = 'cfg/roidepth_0_0_2.cfg'
    img_size: List[int] = [128, 128]
    weights: str        = 'weights/roi_net_1_0_0_pre_1000000.weights'
    device: str         = '0'
    epochs: int         = 1
    batch_size: int     = 64
    data: str           = 'data/roidepth-kitti.data'
    multi_scale: bool   = True
    rect: bool          = False
    resume: bool        = True
    nosace: bool        = True
    notest: bool        = True
    evolve: bool        = True
    bucket: str         = ''
    cache_imgs: bool    = True
    name: str           = ''
    adam: bool          = True
    single_cls: bool    = True
    freeze_layers: bool = True
    conf_thres: float   = 0.001
    iou_thres: float    = 0.35
    save_json: bool     = True
    test_mode: int      = 'test'
    augment: bool       = True
    names: str          = 'data/cls5.names'
    source: str         = 'dataset/kitti/test/images'
    output: str         = 'output'
    fourcc: str         = 'mp4v'
    half: bool          = True
    view_img: bool      = True
    save_txt: bool      = True
    classes: Any        = None
    agnostic_nms: bool  = True

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

        log.info(vars(self))
        check_file(self.cfg)
        check_file(self.data)
        mixed_prec = self.check_apex()
        device = select_device(self.device, apex=mixed_prec, batch_size=self.batch_size)
        if device.type == 'cpu':
            mixed_prec = False
        if self.evolve:
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination
                # Mutate
                method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                npr = np.random
                npr.seed(int(time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                if method == 1:
                    v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                elif method == 3:
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(HYPER_PARAMS.keys()):  # plt.hist(v.ravel(), 300)
                    HYPER_PARAMS[k] = x[i + 7] * v[i]  # mutate
            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                HYPER_PARAMS[k] = np.clip(HYPER_PARAMS[k], v[0], v[1])
            # Train mutation
            results = self.run()
            # Write mutation results
            print_mutation(HYPER_PARAMS, results, self.bucket)
            # Plot results
            plot_evolution_results(HYPER_PARAMS)
        else:  # Train normally
            log.info('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(comment=self.name)
            self.run(HYPER_PARAMS)  # train normally

    def run(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        # self.check_args()

    def check_apex(self):
        mixed_precision = True
        try:  
            from apex import amp
        except:
            log.info(
                'Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex'
            )
            mixed_precision = False  # not installed
        return mixed_precision

    def check_args(self):
        """TODO

        Args:
            TODO

        Raises:
            TODO

        Returns:
            TODO
        """

        # Extend to 3 sizes (min, max, test)
        self.img_size.extend([self.img_size[-1]] * (3 - len(self.img_size)))  
        grid_size = 32
        assert fmod(self.img_size[0], grid_size) ==  0, \
            'img_size {} must be a {}-multiple'.format(self.img_size, grid_size)
        self.multi_scale |= self.img_size[0] != self.img_size[1]  
        if self.multi_scale:
            if self.img_size[0] == self.img_size[1]:
                self.img_size[0] //= 1.5
                self.img_size[1] //= 0.667
            grid_min, grid_max = self.img_size[0] // grid_size, self.img_size[1] // grid_size
            self.img_size[0], self.img_size[1] = \
                int(grid_min * grid_size), int(grid_max * grid_size)
        return 

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
