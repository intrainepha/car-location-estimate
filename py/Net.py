import torch
from torch import nn
from torch import Tensor
from torch.nn.common_types import _size_2_t
from typing import List

class ROIEstNet(nn.Module):
    """
    Location estimation based on Region of Interesrt(ROI) network. 

    Args:
        model_config (str): Model configuration file path.
        image_size (tuple, optional): Image size. Default: (416, 416).

    Attributes:
        model_config (str): Model configuration file path.
        image_size (tuple, optional): Image size. Default: (416, 416).

    """
    def __init__(self, image_size:list = [416, 416], gray:bool = False):
        """Build network base on config.

        Args:
            image_size (tuple, optional): Image size. Default: (416, 416).

        Raises:
            None

        Returns:
            None
        """
        super(ROIEstNet, self).__init__()
        self.image_size = image_size
        self.module_list = nn.ModuleList()
        self.output_filters = [3] if not gray else [1]
        self._build()

    def _build(self):
        """Build torch network.

        Args:
            None

        Raises:
            None

        Returns:
            None
        """
        # Layer 0, (128,128,3)->(128,128,8)->(64,64,8)
        self._conv2d(
            filters=8, kernel_size=3, stride=1, padding=1, 
            maxpool_kernel_size=2, maxpool_stride=2, maxpool_padding=(2-1)//2
        )
        # Layer 1, (64,64,8)->(64,64,8)->(32,32,8)
        self._conv2d(
            filters=8, kernel_size=3, stride=1, padding=1, 
            maxpool_kernel_size=2, maxpool_stride=2, maxpool_padding=(2-1)//2
        )
        # Layer 2, (32,32,8)->(32,32,32)->(32,32,32)->(32,32,32) 
        self._residual(filters=32, kernel_size=3, down_sample=False)
        # Layer 3, (32,32,32)->(16,16,48)->(16,16,48)->(16,16,48) 
        self._residual(filters=48, kernel_size=3, down_sample=True)
        # Layer 4, (16,16,48)->(8,8,64)->(8,8,64)->(8,8,64) 
        self._residual(filters=64, kernel_size=3, down_sample=True)
        # Layer 5, (8,8,64)->(4,4,128)->(4,4,128)->(4,4,128) 
        self._residual(filters=128, kernel_size=3, down_sample=True)
        # Layer 6, (4,4,128)->(2,2,96)->(2,2,96)->(2,2,96) 
        self._residual(filters=96, kernel_size=3, down_sample=True)
        # >>>>>>>>>>>>>>>>>>>>>>>> Upsample <<<<<<<<<<<<<<<<<<<<<<<<<
        # Layer 7 
        self._route(layers=[5])
        # Layer 8, (16,16,128)
        self._upsample(stride=4)
        # Layer 9 
        self._route(layers=[3, 8])
        # Layer 10, (16,16,32)
        self._conv2d(filters=32, kernel_size=3, stride=1, padding=1)
        # Layer 11, (32,32,32)
        self._upsample(stride=2)
        # Layer 12, (32,32,64)
        self._route(layers=[2,11])
        # >>>>>>>>>>>>>>>>>>>>>>>> Depth regression head <<<<<<<<<<<<<<<<<<<<<<<<
        # Layer 13, (32,32,48)
        self._conv2d(filters=48, kernel_size=3, stride=1, padding=1)
        # Layer 14, (16,16,32)
        self._conv2d(filters=48, kernel_size=3, stride=2, padding=1)
        # Layer 15, (16,16,36)
        self._concat_roi(catends=4)
        # Layer 16, (8,8,64)
        self._conv2d(filters=64, kernel_size=3, stride=2, padding=1)
        # Layer 17, (4,4,128)
        self._conv2d(filters=128, kernel_size=3, stride=2, padding=1)
        # Layer 18, (2,2,150)
        self._conv2d(filters=128, kernel_size=3, stride=2, padding=1)
        # Layer 19, (1,600)
        self._flatten()
        # Layer 20, (1,150)
        self._fully_connnect(in_size=600, out_size=150)
        # Layer 21, (1,1)
        self._fully_connnect(in_size=150, out_size=1)
        # >>>>>>>>>>>>>>>>>>>>>>>> YOLO detection head <<<<<<<<<<<<<<<<<<<<<<<<
        # Layer 22, (2,2,96)
        self._route(layers=[12])
        # Layer 23, (2,2,64)
        self._conv2d(filters=64, kernel_size=3, stride=1, padding=1)
        # Layer 24, (2,2,45)
        self._conv2d(filters=45, kernel_size=3, stride=1, padding=1)
        # Layer 25
        self._yolo(
            classes=4,
            anchors = [[5, 7] ,[10, 14], [15, 21], [23, 27], [37, 58]]
        )

    def _conv2d(
        self,
        filters: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: _size_2_t,
        maxpool_kernel_size: _size_2_t = None,
        maxpool_stride: _size_2_t = None,
        maxpool_padding: _size_2_t = None
    ) -> None:
        """Build conv2d with batch normalization and maxpool.

        Args:
            TODO

        Raises:
            TODO

        Returns:
            None
        """
        modules = nn.Sequential()
        modules.add_module(
            "Conv2d", 
            nn.Conv2d(
                in_channels=self.output_filters[-1], out_channels=filters,
                kernel_size=kernel_size, stride=stride,
                padding=padding, bias=False
            )
        )
        modules.add_module("BatchNorm2d", nn.BatchNorm2d(8, momentum=0.03, eps=1E-4))
        modules.add_module("Activation", nn.LeakyReLU(0.1, True))
        if maxpool_kernel_size and maxpool_stride and maxpool_padding:
            modules.add_module(
                nn.MaxPool2d(
                    kernel_size=maxpool_kernel_size,
                    stride=maxpool_stride,
                    padding=maxpool_padding
                )
            )
        self.module_list.append(modules)
        self.output_filters.append(filters)
        return

    def _residual(
        self,
        filters: int,
        kernel_size: _size_2_t,
        down_sample: bool = False
    ) -> None:
        """Build conv2d with batch normalization and maxpool.

        Args:
            TODO

        Raises:
            TODO

        Returns:
            None
        """
        modules = nn.Sequential()
        modules.add_module(
            "Conv2d", 
            nn.Conv2d(
                in_channels=self.output_filters[-1], out_channels=filters,
                kernel_size=kernel_size, stride=1 if not down_sample else 2,
                padding=1, bias=False
            )
        )
        modules.add_module("BatchNorm2d", nn.BatchNorm2d(8, momentum=0.03, eps=1E-4))
        modules.add_module("Activation", nn.LeakyReLU(0.1, True))
        modules.add_module(
            "Conv2d", 
            nn.Conv2d(
                in_channels=filters, out_channels=filters,
                kernel_size=kernel_size, stride=1,
                padding=1, bias=False
            )
        )
        modules.add_module("BatchNorm2d", nn.BatchNorm2d(8, momentum=0.03, eps=1E-4))
        modules.add_module("Activation", nn.LeakyReLU(0.1, True))
        modules = _WeightedFeatureFusion(layers=[-2], weight=False)
        self.module_list.append(modules)
        self.output_filters.append(filters)
        return

    def _route(self, layers: List[int]) -> None:
        filters = sum(
            [self.output_filters[layer + 1 if layer > 0 else layer] for layer in layers]
        ) #FIXME:layer+1???
        modules = _FeatureConcat(layers=layers)
        self.module_list.append(modules)
        self.output_filters.append(filters)
        return

    def _upsample(self, stride: int) -> None:
        modules = nn.Upsample(scale_factor=stride)
        self.module_list.append(modules)
        self.output_filters.append(self.output_filters[-1])
        return
    
    def _concat_roi(self, catends: int) -> None:
        filters = self.output_filters[-1]+catends
        modules = _Concat(catends=catends)
        self.module_list.append(modules)
        self.output_filters.append(filters)
        return
        
    def _flatten(self) -> None:
        modules = _Flatten()
        self.module_list.append(modules)
        self.output_filters.append(None)
        return
        
    def _fully_connnect(self, in_size:int, out_size:int) -> None:
        modules = FullyConnect(in_size, out_size)
        self.module_list.append(modules)
        self.output_filters.append(out_size)
        return

    def _yolo(
        self,
        classes: int = 1,
        anchors:List[List[int]] = [[5, 7] ,[10, 14], [15, 21], [23, 27], [37, 58]]
    ) -> None:
        stride = [32, 16, 8]  # P5, P4, P3 strides
        modules = _YOLOLayer(
            anchors=anchors,    # anchor list
            num_classes=classes, # number of classes
            stride=stride[0]
        )
        self.module_list.append(modules)
        self.output_filters.append(None)
        return

class _WeightedFeatureFusion(nn.Module):
    def __init__(self, layers: nn.ModuleList, weight: bool = False) -> None:
        """

        Args:
            layers:
            weight:

        """
        super(_WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x: torch.Tensor, outputs: torch.Tensor) -> Tensor:
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]
        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels
            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

class _FeatureConcat(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        """

        Args:
            layers (nn.ModuleList):

        """
        super(_FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat([x[i] for i in self.layers], 1) if self.multiple else x[self.layers[0]]

        return x

class _Concat(nn.Module): # TODO: can _Concat be relapced by _FeatureConcat?
    def __init__(self, catends: int):
        super(_Concat, self).__init__()
        self.cantends = catends

    def forward(self, x:Tensor, y:list=None):
        y = [[0]*self.cantends]*x.shape[0] if not y else y
        y_tensor = torch.ones(x.shape[0], len(y[0]), x.shape[-2], x.shape[-1], device=x.device)
        for i in range(len(y)):
            for j in range(len(y[i])):
                y_tensor[i,j,:,:] = y_tensor[i,j,:,:]*torch.Tensor([y[i][j]]).to(x.device)
        return torch.cat([x, y_tensor], 1)

class _Flatten(nn.Module):
    def __init__(self):
        super(_Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class FullyConnect(nn.Module):
    def __init__(self, n_input, n_output):
        super(FullyConnect, self).__init__()
        self.n_output = n_output
        self.n_input = n_input
        self.linear_func = nn.Linear(n_input, n_output)

    def forward(self, x):
        return torch.sigmoid(self.linear_func(x))
    
class _YOLOLayer(nn.Module):
    def __init__(
            self,
            anchors: list,
            num_classes: int,
            stride: int,
    ) -> None:
        """
        Args:
            anchors (list): List of anchors.
            num_classes (int): Number of classes.
            stride (int): Stride.
        """
        super(_YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride
        self.na = len(anchors)  # number of anchors (3)
        self.num_classes = num_classes  # number of classes (80)
        self.num_classes_output = num_classes + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y grid points
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

    def create_grids(self, ng=(13, 13), device="cpu"):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)
        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)],
                                    indexing="ij")
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()
        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device)
        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.num_classes_output, self.ny, self.nx)
        p = p.permute(0, 1, 3, 4, 2).contiguous()  # prediction
        if self.training:
            return p
        else:  # inference
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.num_classes_output), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]