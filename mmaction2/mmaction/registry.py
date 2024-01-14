# Copyright (c) OpenMMLab. All rights reserved.
"""MMAction provides 20 registry nodes to support using modules across
projects. Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

import itertools
import math
from typing import Iterator, Optional, Sized
import torch
from torch.utils.data import Sampler
from mmengine.dist import get_dist_info, sync_random_seed
from time import time
import os.path as osp
import random
import pandas as pd

import copy
import datetime
import re
from collections import OrderedDict
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
import torch
from typing import Tuple, OrderedDict
from mmengine.device import get_max_cuda_memory, is_cuda_available
import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from mmengine.runner.utils import calc_dynamic_intervals
# from mmengine.runner.runner import runner
from mmengine.registry import LOG_PROCESSORS
from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import FUNCTIONS as MMENGINE_FUNCTION
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import INFERENCERS as MMENGINE_INFERENCERS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    'runner', parent=MMENGINE_RUNNERS, locations=['mmaction.engine.runner'])
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMENGINE_RUNNER_CONSTRUCTORS,
    locations=['mmaction.engine.runner'])
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry(
    'loop', parent=MMENGINE_LOOPS, locations=['mmaction.engine.runner'])
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, locations=['mmaction.engine.hooks'])

# manage data-related modules
DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=['mmaction.datasets'])
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMENGINE_DATA_SAMPLERS,
    locations=['mmaction.datasets'])
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['mmaction.datasets.transforms'])

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model', parent=MMENGINE_MODELS, locations=['mmaction.models'])
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=['mmaction.models'])
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=['mmaction.models'])

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    locations=['mmaction.engine.optimizers'])
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim_wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    locations=['mmaction.engine.optimizers'])
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=['mmaction.engine.optimizers'])
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    locations=['mmaction.engine'])

# manage all kinds of metrics
METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, locations=['mmaction.evaluation'])
# manage evaluator
EVALUATOR = Registry(
    'evaluator', parent=MMENGINE_EVALUATOR, locations=['mmaction.evaluation'])

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', parent=MMENGINE_TASK_UTILS, locations=['mmaction.models'])

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=MMENGINE_VISUALIZERS,
    locations=['mmaction.visualization'])
# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMENGINE_VISBACKENDS,
    locations=['mmaction.visualization'])

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=MMENGINE_LOG_PROCESSORS,
    locations=['mmaction.engine'])

# manage inferencer
INFERENCERS = Registry(
    'inferencer',
    parent=MMENGINE_INFERENCERS,
    locations=['mmaction.apis.inferencers'])

# manage function
FUNCTION = Registry(
    'function', parent=MMENGINE_FUNCTION, locations=['mmaction.mmengine'])

# Tokenizer to encode sequence
TOKENIZER = Registry(
    'tokenizer',
    locations=['mmaction.models'],
)



# # Copyright (c) OpenMMLab. All rights reserved.
# from abc import abstractmethod
# from collections import OrderedDict
# from typing import Dict, Optional, Tuple, Union

# import torch
# import torch.nn as nn

# from mmengine.optim import OptimWrapper
# from mmengine.utils import is_list_of
# from mmengine.model.base_module import BaseModule
# from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor

# import copy
# import logging
# import os
# import os.path as osp
# import pickle
# import platform
# import time
# import warnings
# from collections import OrderedDict
# from functools import partial
# from typing import Callable, Dict, List, Optional, Sequence, Union

# import torch
# import torch.nn as nn
# from torch.nn.parallel.distributed import DistributedDataParallel
# from torch.optim import Optimizer
# from torch.utils.data import DataLoader

# import mmengine
# from mmengine.config import Config, ConfigDict
# from mmengine.dataset import worker_init_fn as default_worker_init_fn
# from mmengine.device import get_device
# from mmengine.dist import (broadcast, get_dist_info, get_rank, get_world_size,
#                            init_dist, is_distributed, master_only)
# from mmengine.evaluator import Evaluator
# from mmengine.fileio import FileClient, join_path
# from mmengine.hooks import Hook
# from mmengine.logging import MessageHub, MMLogger, print_log
# from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
#                             is_model_wrapper, revert_sync_batchnorm)
# from mmengine.model.efficient_conv_bn_eval import \
#     turn_on_efficient_conv_bn_eval
# from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
#                             build_optim_wrapper)
# from mmengine.registry import (FUNCTIONS, DefaultScope)
# from mmengine.utils import apply_to, digit_version, get_git_hash, is_seq_of
# from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
#                                      set_multi_processing)
# from mmengine.visualization import Visualizer
# from mmengine.runner.activation_checkpointing import turn_on_activation_checkpointing
# from mmengine.runner.base_loop import BaseLoop
# from mmengine.runner.checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
#                          find_latest_checkpoint, save_checkpoint,
#                          weights_to_cpu)
# from mmengine.runner.log_processor import LogProcessor
# from mmengine.runner.loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
# from mmengine.runner.priority import Priority, get_priority
# from mmengine.runner.utils import _get_batch_size, set_random_seed
# class BaseModel(BaseModule):
#     """Base class for all algorithmic models.

#     BaseModel implements the basic functions of the algorithmic model, such as
#     weights initialize, batch inputs preprocess(see more information in
#     :class:`BaseDataPreprocessor`), parse losses, and update model parameters.

#     Subclasses inherit from BaseModel only need to implement the forward
#     method, which implements the logic to calculate loss and predictions,
#     then can be trained in the runner.

#     Examples:
#         >>> @MODELS.register_module()
#         >>> class ToyModel(BaseModel):
#         >>>
#         >>>     def __init__(self):
#         >>>         super().__init__()
#         >>>         self.backbone = nn.Sequential()
#         >>>         self.backbone.add_module('conv1', nn.Conv2d(3, 6, 5))
#         >>>         self.backbone.add_module('pool', nn.MaxPool2d(2, 2))
#         >>>         self.backbone.add_module('conv2', nn.Conv2d(6, 16, 5))
#         >>>         self.backbone.add_module('fc1', nn.Linear(16 * 5 * 5, 120))
#         >>>         self.backbone.add_module('fc2', nn.Linear(120, 84))
#         >>>         self.backbone.add_module('fc3', nn.Linear(84, 10))
#         >>>
#         >>>         self.criterion = nn.CrossEntropyLoss()
#         >>>
#         >>>     def forward(self, batch_inputs, data_samples, mode='tensor'):
#         >>>         data_samples = torch.stack(data_samples)
#         >>>         if mode == 'tensor':
#         >>>             return self.backbone(batch_inputs)
#         >>>         elif mode == 'predict':
#         >>>             feats = self.backbone(batch_inputs)
#         >>>             predictions = torch.argmax(feats, 1)
#         >>>             return predictions
#         >>>         elif mode == 'loss':
#         >>>             feats = self.backbone(batch_inputs)
#         >>>             loss = self.criterion(feats, data_samples)
#         >>>             return dict(loss=loss)

#     Args:
#         data_preprocessor (dict, optional): The pre-process config of
#             :class:`BaseDataPreprocessor`.
#         init_cfg (dict, optional): The weight initialized config for
#             :class:`BaseModule`.

#     Attributes:
#         data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
#             pre-processing data sampled by dataloader to the format accepted by
#             :meth:`forward`.
#         init_cfg (dict, optional): Initialization config dict.
#     """

#     def __init__(self,
#                  data_preprocessor: Optional[Union[dict, nn.Module]] = None,
#                  init_cfg: Optional[dict] = None):
#         super().__init__(init_cfg)
#         if data_preprocessor is None:
#             data_preprocessor = dict(type='BaseDataPreprocessor')
#         if isinstance(data_preprocessor, nn.Module):
#             self.data_preprocessor = data_preprocessor
#         elif isinstance(data_preprocessor, dict):
#             self.data_preprocessor = MODELS.build(data_preprocessor)
#         else:
#             raise TypeError('data_preprocessor should be a `dict` or '
#                             f'`nn.Module` instance, but got '
#                             f'{type(data_preprocessor)}')

#     def train_step(self, data: Union[dict, tuple, list],
#                    optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
#         """Implements the default model training process including
#         preprocessing, model forward propagation, loss calculation,
#         optimization, and back-propagation.

#         During non-distributed training. If subclasses do not override the
#         :meth:`train_step`, :class:`EpochBasedTrainLoop` or
#         :class:`IterBasedTrainLoop` will call this method to update model
#         parameters. The default parameter update process is as follows:

#         1. Calls ``self.data_processor(data, training=False)`` to collect
#            batch_inputs and corresponding data_samples(labels).
#         2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
#            loss
#         3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
#            backward and dict of loss tensor used to log messages.
#         4. Calls ``optim_wrapper.update_params(loss)`` to update model.

#         Args:
#             data (dict or tuple or list): Data sampled from dataset.
#             optim_wrapper (OptimWrapper): OptimWrapper instance
#                 used to update model parameters.

#         Returns:
#             Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
#         """
#         # Enable automatic mixed precision training context.
#         with optim_wrapper.optim_context(self):
#             data = self.data_preprocessor(data, True)
#             losses = self._run_forward(data, mode='loss')  # type: ignore
#         parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
#         optim_wrapper.update_params(parsed_losses)
#         return log_vars

#     def val_step(self, data: Union[tuple, dict, list],optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
#         """Gets the predictions of given data.

#         Calls ``self.data_preprocessor(data, False)`` and
#         ``self(inputs, data_sample, mode='predict')`` in order. Return the
#         predictions which will be passed to evaluator.

#         Args:
#             data (dict or tuple or list): Data sampled from dataset.

#         Returns:
#             list: The predictions of given data.
#         """
#         with optim_wrapper.optim_context(self):
#             data = self.data_preprocessor(data, True)
#             losses = self._run_forward(data, mode='loss')  # type: ignore
#             # print("1____",losses)
#         parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
#         # print("2____",parsed_losses)
#         optim_wrapper.update_params(parsed_losses)
#         # print("3____",log_vars, self._run_forward(data, mode='predict'))
#         return log_vars, self._run_forward(data, mode='predict')  # type: ignore

#     def test_step(self, data: Union[dict, tuple, list]) -> list:
#         """``BaseModel`` implements ``test_step`` the same as ``val_step``.

#         Args:
#             data (dict or tuple or list): Data sampled from dataset.

#         Returns:
#             list: The predictions of given data.
#         """
#         data = self.data_preprocessor(data, False)
#         return self._run_forward(data, mode='predict')  # type: ignore

#     def parse_losses(
#         self, losses: Dict[str, torch.Tensor]
#     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         """Parses the raw outputs (losses) of the network.

#         Args:
#             losses (dict): Raw output of the network, which usually contain
#                 losses and other necessary information.

#         Returns:
#             tuple[Tensor, dict]: There are two elements. The first is the
#             loss tensor passed to optim_wrapper which may be a weighted sum
#             of all losses, and the second is log_vars which will be sent to
#             the logger.
#         """
#         log_vars = []
#         for loss_name, loss_value in losses.items():
#             if isinstance(loss_value, torch.Tensor):
#                 log_vars.append([loss_name, loss_value.mean()])
#             elif is_list_of(loss_value, torch.Tensor):
#                 log_vars.append(
#                     [loss_name,
#                      sum(_loss.mean() for _loss in loss_value)])
#             else:
#                 raise TypeError(
#                     f'{loss_name} is not a tensor or list of tensors')

#         loss = sum(value for key, value in log_vars if 'loss' in key)
#         log_vars.insert(0, ['loss', loss])
#         log_vars = OrderedDict(log_vars)  # type: ignore

#         return loss, log_vars  # type: ignore

#     def to(self, *args, **kwargs) -> nn.Module:
#         """Overrides this method to call :meth:`BaseDataPreprocessor.to`
#         additionally.

#         Returns:
#             nn.Module: The model itself.
#         """

#         # Since Torch has not officially merged
#         # the npu-related fields, using the _parse_to function
#         # directly will cause the NPU to not be found.
#         # Here, the input parameters are processed to avoid errors.
#         if args and isinstance(args[0], str) and 'npu' in args[0]:
#             import torch_npu
#             args = tuple([
#                 list(args)[0].replace(
#                     'npu', torch_npu.npu.native_device if hasattr(
#                         torch_npu.npu, 'native_device') else 'privateuseone')
#             ])
#         if kwargs and 'npu' in str(kwargs.get('device', '')):
#             import torch_npu
#             kwargs['device'] = kwargs['device'].replace(
#                 'npu', torch_npu.npu.native_device if hasattr(
#                     torch_npu.npu, 'native_device') else 'privateuseone')

#         device = torch._C._nn._parse_to(*args, **kwargs)[0]
#         if device is not None:
#             self._set_device(torch.device(device))
#         return super().to(*args, **kwargs)

#     def cuda(
#         self,
#         device: Optional[Union[int, str, torch.device]] = None,
#     ) -> nn.Module:
#         """Overrides this method to call :meth:`BaseDataPreprocessor.cuda`
#         additionally.

#         Returns:
#             nn.Module: The model itself.
#         """
#         if device is None or isinstance(device, int):
#             device = torch.device('cuda', index=device)
#         self._set_device(torch.device(device))
#         return super().cuda(device)

#     def mlu(
#         self,
#         device: Union[int, str, torch.device, None] = None,
#     ) -> nn.Module:
#         """Overrides this method to call :meth:`BaseDataPreprocessor.mlu`
#         additionally.

#         Returns:
#             nn.Module: The model itself.
#         """
#         device = torch.device('mlu', torch.mlu.current_device())
#         self._set_device(device)
#         return super().mlu()

#     def npu(
#         self,
#         device: Union[int, str, torch.device, None] = None,
#     ) -> nn.Module:
#         """Overrides this method to call :meth:`BaseDataPreprocessor.npu`
#         additionally.

#         Returns:
#             nn.Module: The model itself.

#         Note:
#             This generation of NPU(Ascend910) does not support
#             the use of multiple cards in a single process,
#             so the index here needs to be consistent with the default device
#         """
#         device = torch.npu.current_device()
#         self._set_device(device)
#         return super().npu()

#     def cpu(self, *args, **kwargs) -> nn.Module:
#         """Overrides this method to call :meth:`BaseDataPreprocessor.cpu`
#         additionally.

#         Returns:
#             nn.Module: The model itself.
#         """
#         self._set_device(torch.device('cpu'))
#         return super().cpu()

#     def _set_device(self, device: torch.device) -> None:
#         """Recursively set device for `BaseDataPreprocessor` instance.

#         Args:
#             device (torch.device): the desired device of the parameters and
#                 buffers in this module.
#         """

#         def apply_fn(module):
#             if not isinstance(module, BaseDataPreprocessor):
#                 return
#             if device is not None:
#                 module._device = device

#         self.apply(apply_fn)

#     @abstractmethod
#     def forward(self,
#                 inputs: torch.Tensor,
#                 data_samples: Optional[list] = None,
#                 mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
#         """Returns losses or predictions of training, validation, testing, and
#         simple inference process.

#         ``forward`` method of BaseModel is an abstract method, its subclasses
#         must implement this method.

#         Accepts ``batch_inputs`` and ``data_sample`` processed by
#         :attr:`data_preprocessor`, and returns results according to mode
#         arguments.

#         During non-distributed training, validation, and testing process,
#         ``forward`` will be called by ``BaseModel.train_step``,
#         ``BaseModel.val_step`` and ``BaseModel.test_step`` directly.

#         During distributed data parallel training process,
#         ``MMSeparateDistributedDataParallel.train_step`` will first call
#         ``DistributedDataParallel.forward`` to enable automatic
#         gradient synchronization, and then call ``forward`` to get training
#         loss.

#         Args:
#             inputs (torch.Tensor): batch input tensor collated by
#                 :attr:`data_preprocessor`.
#             data_samples (list, optional):
#                 data samples collated by :attr:`data_preprocessor`.
#             mode (str): mode should be one of ``loss``, ``predict`` and
#                 ``tensor``

#                 - ``loss``: Called by ``train_step`` and return loss ``dict``
#                   used for logging
#                 - ``predict``: Called by ``val_step`` and ``test_step``
#                   and return list of results used for computing metric.
#                 - ``tensor``: Called by custom use to get ``Tensor`` type
#                   results.

#         Returns:
#             dict or list:
#                 - If ``mode == loss``, return a ``dict`` of loss tensor used
#                   for backward and logging.
#                 - If ``mode == predict``, return a ``list`` of inference
#                   results.
#                 - If ``mode == tensor``, return a tensor or ``tuple`` of tensor
#                   or ``dict`` of tensor for custom use.
#         """

#     def _run_forward(self, data: Union[dict, tuple, list],
#                      mode: str) -> Union[Dict[str, torch.Tensor], list]:
#         """Unpacks data for :meth:`forward`

#         Args:
#             data (dict or tuple or list): Data sampled from dataset.
#             mode (str): Mode of forward.

#         Returns:
#             dict or list: Results of training or testing mode.
#         """
#         if isinstance(data, dict):
#             results = self(**data, mode=mode)
#         elif isinstance(data, (list, tuple)):
#             results = self(*data, mode=mode)
#         else:
#             raise TypeError('Output of `data_preprocessor` should be '
#                             f'list, tuple or dict, but got {type(data)}')
#         return results



# ConfigType = Union[Dict, Config, ConfigDict]
# ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
#                                                        List[_ParamScheduler]]]
# OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]


# class _SlicedDataset:

#     def __init__(self, dataset, length) -> None:
#         self._dataset = dataset
#         self._length = length

#     def __getattr__(self, name):
#         return getattr(self._dataset, name)

#     def __getitem__(self, idx):
#         return self._dataset[idx]

#     def __len__(self):
#         return self._length


# @RUNNERS.register_module()
# class Runner:
#     """A training helper for PyTorch.

#     Runner object can be built from config by ``runner = Runner.from_cfg(cfg)``
#     where the ``cfg`` usually contains training, validation, and test-related
#     configurations to build corresponding components. We usually use the
#     same config to launch training, testing, and validation tasks. However,
#     only some of these components are necessary at the same time, e.g.,
#     testing a model does not need training or validation-related components.

#     To avoid repeatedly modifying config, the construction of ``Runner`` adopts
#     lazy initialization to only initialize components when they are going to be
#     used. Therefore, the model is always initialized at the beginning, and
#     training, validation, and, testing related components are only initialized
#     when calling ``runner.train()``, ``runner.val()``, and ``runner.test()``,
#     respectively.

#     Args:
#         model (:obj:`torch.nn.Module` or dict): The model to be run. It can be
#             a dict used for build a model.
#         work_dir (str): The working directory to save checkpoints. The logs
#             will be saved in the subdirectory of `work_dir` named
#             :attr:`timestamp`.
#         train_dataloader (Dataloader or dict, optional): A dataloader object or
#             a dict to build a dataloader. If ``None`` is given, it means
#             skipping training steps. Defaults to None.
#             See :meth:`build_dataloader` for more details.
#         val_dataloader (Dataloader or dict, optional): A dataloader object or
#             a dict to build a dataloader. If ``None`` is given, it means
#             skipping validation steps. Defaults to None.
#             See :meth:`build_dataloader` for more details.
#         test_dataloader (Dataloader or dict, optional): A dataloader object or
#             a dict to build a dataloader. If ``None`` is given, it means
#             skipping test steps. Defaults to None.
#             See :meth:`build_dataloader` for more details.
#         train_cfg (dict, optional): A dict to build a training loop. If it does
#             not provide "type" key, it should contain "by_epoch" to decide
#             which type of training loop :class:`EpochBasedTrainLoop` or
#             :class:`IterBasedTrainLoop` should be used. If ``train_cfg``
#             specified, :attr:`train_dataloader` should also be specified.
#             Defaults to None. See :meth:`build_train_loop` for more details.
#         val_cfg (dict, optional): A dict to build a validation loop. If it does
#             not provide "type" key, :class:`ValLoop` will be used by default.
#             If ``val_cfg`` specified, :attr:`val_dataloader` should also be
#             specified. If ``ValLoop`` is built with `fp16=True``,
#             ``runner.val()`` will be performed under fp16 precision.
#             Defaults to None. See :meth:`build_val_loop` for more details.
#         test_cfg (dict, optional): A dict to build a test loop. If it does
#             not provide "type" key, :class:`TestLoop` will be used by default.
#             If ``test_cfg`` specified, :attr:`test_dataloader` should also be
#             specified. If ``ValLoop`` is built with `fp16=True``,
#             ``runner.val()`` will be performed under fp16 precision.
#             Defaults to None. See :meth:`build_test_loop` for more details.
#         auto_scale_lr (dict, Optional): Config to scale the learning rate
#             automatically. It includes ``base_batch_size`` and ``enable``.
#             ``base_batch_size`` is the batch size that the optimizer lr is
#             based on. ``enable`` is the switch to turn on anval_loopd off the feature.
#         optim_wrapper (OptimWrapper or dict, optional):
#             Computing gradient of model parameters. If specified,
#             :attr:`train_dataloader` should also be specified. If automatic
#             mixed precision or gradient accmulation
#             training is required. The type of ``optim_wrapper`` should be
#             AmpOptimizerWrapper. See :meth:`build_optim_wrapper` for
#             examples. Defaults to None.
#         param_scheduler (_ParamScheduler or dict or list, optional):
#             Parameter scheduler for updating optimizer parameters. If
#             specified, :attr:`optimizer` should also be specified.
#             Defaults to None.
#             See :meth:`build_param_scheduler` for examples.
#         val_evaluator (Evaluator or dict or list, optional): A evaluator object
#             used for computing metrics for validation. It can be a dict or a
#             list of dict to build a evaluator. If specified,
#             :attr:`val_dataloader` should also be specified. Defaults to None.
#         test_evaluator (Evaluator or dict or list, optional): A evaluator
#             object used for computing metrics for test steps. It can be a dict
#             or a list of dict to build a evaluator. If specified,
#             :attr:`test_dataloader` should also be specified. Defaults to None.
#         default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks to
#             execute default actions like updating model parameters and saving
#             checkpoints. Default hooks are ``OptimizerHook``,
#             ``IterTimerHook``, ``LoggerHook``, ``ParamSchedulerHook`` and
#             ``CheckpointHook``. Defaults to None.
#             See :meth:`register_default_hooks` for more details.
#         custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
#             custom actions like visualizing images processed by pipeline.
#             Defaults to None.
#         data_preprocessor (dict, optional): The pre-process config of
#             :class:`BaseDataPreprocessor`. If the ``model`` argument is a dict
#             and doesn't contain the key ``data_preprocessor``, set the argument
#             as the ``data_preprocessor`` of the ``model`` dict.
#             Defaults to None.
#         load_from (str, optional): The checkpoint file to load from.
#             Defaults to None.
#         resume (bool): Whether to resume training. Defaults to False. If
#             ``resume`` is True and ``load_from`` is None, automatically to
#             find latest checkpoint from ``work_dir``. If not found, resuming
#             does nothing.
#         launcher (str): Way to launcher multi-process. Supported launchers
#             are 'pytorch', 'mpi', 'slurm' and 'none'. If 'none' is provided,
#             non-distributed environment will be launched.
#         env_cfg (dict): A dict used for setting environment. Defaults to
#             dict(dist_cfg=dict(backend='nccl')).
#         log_processor (dict, optional): A processor to format logs. Defaults to
#             None.
#         log_level (int or str): The log level of MMLogger handlers.
#             Defaults to 'INFO'.
#         visualizer (Visualizer or dict, optional): A Visualizer object or a
#             dict build Visualizer object. Defaults to None. If not
#             specified, default config will be used.
#         default_scope (str): Used to reset registries location.
#             Defaults to "mmengine".
#         randomness (dict): Some settings to make the experiment as reproducible
#             as possible like seed and deterministic.
#             Defaults to ``dict(seed=None)``. If seed is None, a random number
#             will be generated and it will be broadcasted to all other processes
#             if in distributed environment. If ``cudnn_benchmark`` is
#             ``True`` in ``env_cfg`` but ``deterministic`` is ``True`` in
#             ``randomness``, the value of ``torch.backends.cudnn.benchmark``
#             will be ``False`` finally.
#         experiment_name (str, optional): Name of current experiment. If not
#             specified, timestamp will be used as ``experiment_name``.
#             Defaults to None.
#         cfg (dict or Configdict or :obj:`Config`, optional): Full config.
#             Defaults to None.

#     Note:
#         Since PyTorch 2.0.0, you can enable ``torch.compile`` by passing in
#         `cfg.compile = True`. If you want to control compile options, you
#         can pass a dict, e.g. ``cfg.compile = dict(backend='eager')``.
#         Refer to `PyTorch API Documentation <https://pytorch.org/docs/
#         master/generated/torch.compile.html#torch.compile>`_ for more valid
#         options.

#     Examples:
#         >>> from mmengine.runner import Runner
#         >>> cfg = dict(
#         >>>     model=dict(type='ToyModel'),
#         >>>     work_dir='path/of/work_dir',
#         >>>     train_dataloader=dict(
#         >>>     dataset=dict(type='ToyDataset'),
#         >>>     sampler=dict(type='DefaultSampler', shuffle=True),
#         >>>     batch_size=1,
#         >>>     num_workers=0),
#         >>>     val_dataloader=dict(
#         >>>         dataset=dict(type='ToyDataset'),
#         >>>         sampler=dict(type='DefaultSampler', shuffle=False),
#         >>>        batch_size=1,
#         >>>        num_workers=0),
#         >>>     test_dataloader=dict(
#         >>>         dataset=dict(type='ToyDataset'),
#         >>>         sampler=dict(type='DefaultSampler', shuffle=False),
#         >>>         batch_size=1,
#         >>>         num_workers=0),
#         >>>     auto_scale_lr=dict(base_batch_size=16, enable=False),
#         >>>     optim_wrapper=dict(type='OptimizerWrapper', optimizer=dict(
#         >>>         type='SGD', lr=0.01)),
#         >>>     param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
#         >>>     val_evaluator=dict(type='ToyEvaluator'),
#         >>>     test_evaluator=dict(type='ToyEvaluator'),
#         >>>     train_cfg=dict(by_epoch=True, max_epochs=3, val_interval=1),
#         >>>     val_cfg=dict(),
#         >>>     test_cfg=dict(),
#         >>>     custom_hooks=[],
#         >>>     default_hooks=dict(
#         >>>         timer=dict(type='IterTimerHook'),
#         >>>         checkpoint=dict(type='CheckpointHook', interval=1),
#         >>>         logger=dict(type='LoggerHook'),
#         >>>         optimizer=dict(type='OptimizerHook', grad_clip=False),
#         >>>         param_scheduler=dict(type='ParamSchedulerHook')),
#         >>>     launcher='none',
#         >>>     env_cfg=dict(dist_cfg=dict(backend='nccl')),
#         >>>     log_processor=dict(window_size=20),
#         >>>     visualizer=dict(type='Visualizer',
#         >>>     vis_backends=[dict(type='LocalVisBackend',
#         >>>                        save_dir='temp_dir')])
#         >>>    )
#         >>> runner = Runner.from_cfg(cfg)
#         >>> runner.train()
#         >>> runner.test()
#     """
#     cfg: Config
#     _train_loop: Optional[Union[BaseLoop, Dict]]
#     _val_loop: Optional[Union[BaseLoop, Dict]]
#     _test_loop: Optional[Union[BaseLoop, Dict]]

#     def __init__(
#         self,
#         model: Union[nn.Module, Dict],
#         work_dir: str,
#         train_dataloader: Optional[Union[DataLoader, Dict]] = None,
#         val_dataloader: Optional[Union[DataLoader, Dict]] = None,
#         test_dataloader: Optional[Union[DataLoader, Dict]] = None,
#         train_cfg: Optional[Dict] = None,
#         val_cfg: Optional[Dict] = None,
#         test_cfg: Optional[Dict] = None,
#         auto_scale_lr: Optional[Dict] = None,
#         optim_wrapper: Optional[Union[OptimWrapper, Dict]] = None,
#         param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
#         val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
#         test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
#         default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
#         custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
#         data_preprocessor: Union[nn.Module, Dict, None] = None,
#         load_from: Optional[str] = None,
#         resume: bool = False,
#         launcher: str = 'none',
#         env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
#         log_processor: Optional[Dict] = None,
#         log_level: str = 'INFO',
#         visualizer: Optional[Union[Visualizer, Dict]] = None,
#         default_scope: str = 'mmengine',
#         randomness: Dict = dict(seed=None),
#         experiment_name: Optional[str] = None,
#         cfg: Optional[ConfigType] = None,
#     ):
#         self._work_dir = osp.abspath(work_dir)
#         mmengine.mkdir_or_exist(self._work_dir)

#         # recursively copy the `cfg` because `self.cfg` will be modified
#         # everywhere.
#         if cfg is not None:
#             if isinstance(cfg, Config):
#                 self.cfg = copy.deepcopy(cfg)
#             elif isinstance(cfg, dict):
#                 self.cfg = Config(cfg)
#         else:
#             self.cfg = Config(dict())

#         # lazy initialization
#         training_related = [train_dataloader, train_cfg, optim_wrapper]
#         if not (all(item is None for item in training_related)
#                 or all(item is not None for item in training_related)):
#             raise ValueError(
#                 'train_dataloader, train_cfg, and optim_wrapper should be '
#                 'either all None or not None, but got '
#                 f'train_dataloader={train_dataloader}, '
#                 f'train_cfg={train_cfg}, '
#                 f'optim_wrapper={optim_wrapper}.')
#         self._train_dataloader = train_dataloader
#         self._train_loop = train_cfg

#         self.optim_wrapper: Optional[Union[OptimWrapper, dict]]
#         self.optim_wrapper = optim_wrapper

#         self.auto_scale_lr = auto_scale_lr

#         # If there is no need to adjust learning rate, momentum or other
#         # parameters of optimizer, param_scheduler can be None
#         if param_scheduler is not None and self.optim_wrapper is None:
#             raise ValueError(
#                 'param_scheduler should be None when optim_wrapper is None, '
#                 f'but got {param_scheduler}')

#         # Parse `param_scheduler` to a list or a dict. If `optim_wrapper` is a
#         # `dict` with single optimizer, parsed param_scheduler will be a
#         # list of parameter schedulers. If `optim_wrapper` is
#         # a `dict` with multiple optimizers, parsed `param_scheduler` will be
#         # dict with multiple list of parameter schedulers.
#         self._check_scheduler_cfg(param_scheduler)
#         self.param_schedulers = param_scheduler

#         val_related = [val_dataloader, val_cfg, val_evaluator]
#         if not (all(item is None
#                     for item in val_related) or all(item is not None
#                                                     for item in val_related)):
#             raise ValueError(
#                 'val_dataloader, val_cfg, and val_evaluator should be either '
#                 'all None or not None, but got '
#                 f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
#                 f'val_evaluator={val_evaluator}')
#         self._val_dataloader = val_dataloader
#         self._val_loop = val_cfg
#         self._val_evaluator = val_evaluator

#         test_related = [test_dataloader, test_cfg, test_evaluator]
#         if not (all(item is None for item in test_related)
#                 or all(item is not None for item in test_related)):
#             raise ValueError(
#                 'test_dataloader, test_cfg, and test_evaluator should be '
#                 'either all None or not None, but got '
#                 f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
#                 f'test_evaluator={test_evaluator}')
#         self._test_dataloader = test_dataloader
#         self._test_loop = test_cfg
#         self._test_evaluator = test_evaluator

#         self._launcher = launcher
#         if self._launcher == 'none':
#             self._distributed = False
#         else:
#             self._distributed = True

#         # self._timestamp will be set in the `setup_env` method. Besides,
#         # it also will initialize multi-process and (or) distributed
#         # environment.
#         self.setup_env(env_cfg)
#         # self._deterministic and self._seed will be set in the
#         # `set_randomness`` method
#         self._randomness_cfg = randomness
#         self.set_randomness(**randomness)

#         if experiment_name is not None:
#             self._experiment_name = f'{experiment_name}_{self._timestamp}'
#         elif self.cfg.filename is not None:
#             filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
#             self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
#         else:
#             self._experiment_name = self.timestamp
#         self._log_dir = osp.join(self.work_dir, self.timestamp)
#         mmengine.mkdir_or_exist(self._log_dir)
#         # Used to reset registries location. See :meth:`Registry.build` for
#         # more details.
#         if default_scope is not None:
#             default_scope = DefaultScope.get_instance(  # type: ignore
#                 self._experiment_name,
#                 scope_name=default_scope)
#         self.default_scope = default_scope

#         # Build log processor to format message.
#         log_processor = dict() if log_processor is None else log_processor
#         self.log_processor = self.build_log_processor(log_processor)
#         # Since `get_instance` could return any subclass of ManagerMixin. The
#         # corresponding attribute needs a type hint.
#         self.logger = self.build_logger(log_level=log_level)

#         # Collect and log environment information.
#         self._log_env(env_cfg)

#         # Build `message_hub` for communication among components.
#         # `message_hub` can store log scalars (loss, learning rate) and
#         # runtime information (iter and epoch). Those components that do not
#         # have access to the runner can get iteration or epoch information
#         # from `message_hub`. For example, models can get the latest created
#         # `message_hub` by
#         # `self.message_hub=MessageHub.get_current_instance()` and then get
#         # current epoch by `cur_epoch = self.message_hub.get_info('epoch')`.
#         # See `MessageHub` and `ManagerMixin` for more details.
#         self.message_hub = self.build_message_hub()
#         # visualizer used for writing log or visualizing all kinds of data
#         self.visualizer = self.build_visualizer(visualizer)
#         if self.cfg:
#             self.visualizer.add_config(self.cfg)

#         self._load_from = load_from
#         self._resume = resume
#         # flag to mark whether checkpoint has been loaded or resumed
#         self._has_loaded = False

#         # build a model
#         if isinstance(model, dict) and data_preprocessor is not None:
#             # Merge the data_preprocessor to model config.
#             model.setdefault('data_preprocessor', data_preprocessor)
#         self.model = self.build_model(model)
#         # wrap model
#         self.model = self.wrap_model(
#             self.cfg.get('model_wrapper_cfg'), self.model)

#         # get model name from the model class
#         if hasattr(self.model, 'module'):
#             self._model_name = self.model.module.__class__.__name__
#         else:
#             self._model_name = self.model.__class__.__name__

#         self._hooks: List[Hook] = []
#         # register hooks to `self._hooks`
#         self.register_hooks(default_hooks, custom_hooks)
#         # log hooks information
#         self.logger.info(f'Hooks will be executed in the following '
#                          f'order:\n{self.get_hooks_info()}')

#         # dump `cfg` to `work_dir`
#         self.dump_config()

#     @classmethod
#     def from_cfg(cls, cfg: ConfigType) -> 'Runner':
#         """Build a runner from config.

#         Args:
#             cfg (ConfigType): A config used for building runner. Keys of
#                 ``cfg`` can see :meth:`__init__`.

#         Returns:
#             Runner: A runner build from ``cfg``.
#         """
#         cfg = copy.deepcopy(cfg)
#         runner = cls(
#             model=cfg['model'],
#             work_dir=cfg['work_dir'],
#             train_dataloader=cfg.get('train_dataloader'),
#             val_dataloader=cfg.get('val_dataloader'),
#             test_dataloader=cfg.get('test_dataloader'),
#             train_cfg=cfg.get('train_cfg'),
#             val_cfg=cfg.get('val_cfg'),
#             test_cfg=cfg.get('test_cfg'),
#             auto_scale_lr=cfg.get('auto_scale_lr'),
#             optim_wrapper=cfg.get('optim_wrapper'),
#             param_scheduler=cfg.get('param_scheduler'),
#             val_evaluator=cfg.get('val_evaluator'),
#             test_evaluator=cfg.get('test_evaluator'),
#             default_hooks=cfg.get('default_hooks'),
#             custom_hooks=cfg.get('custom_hooks'),
#             data_preprocessor=cfg.get('data_preprocessor'),
#             load_from=cfg.get('load_from'),
#             resume=cfg.get('resume', False),
#             launcher=cfg.get('launcher', 'none'),
#             env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
#             log_processor=cfg.get('log_processor'),
#             log_level=cfg.get('log_level', 'INFO'),
#             visualizer=cfg.get('visualizer'),
#             default_scope=cfg.get('default_scope', 'mmengine'),
#             randomness=cfg.get('randomness', dict(seed=None)),
#             experiment_name=cfg.get('experiment_name'),
#             cfg=cfg,
#         )

#         return runner

#     @property
#     def experiment_name(self):
#         """str: Name of experiment."""
#         return self._experiment_name

#     @property
#     def model_name(self):
#         """str: Name of the model, usually the module class name."""
#         return self._model_name

#     @property
#     def work_dir(self):
#         """str: The working directory to save checkpoints and logs."""
#         return self._work_dir

#     @property
#     def log_dir(self):
#         return self._log_dir

#     @property
#     def max_epochs(self):
#         """int: Total epochs to train model."""
#         if isinstance(self.train_loop, BaseLoop):
#             return self.train_loop.max_epochs
#         else:
#             return 0

#     @property
#     def max_iters(self):
#         """int: Total iterations to train model."""
#         if isinstance(self.train_loop, BaseLoop):
#             return self.train_loop.max_iters
#         else:
#             return 0

#     @property
#     def epoch(self):
#         """int: Current epoch."""
#         if isinstance(self.train_loop, BaseLoop):
#             return self.train_loop.epoch
#         else:
#             return 0

#     @property
#     def iter(self):
#         """int: Current iteration."""
#         if isinstance(self.train_loop, BaseLoop):
#             return self.train_loop.iter
#         else:
#             return 0

#     @property
#     def launcher(self):
#         """str: Way to launcher multi processes."""
#         return self._launcher

#     @property
#     def distributed(self):
#         """bool: Whether current environment is distributed."""
#         return self._distributed

#     @property
#     def rank(self):
#         """int: Rank of current process."""
#         return self._rank

#     @property
#     def world_size(self):
#         """int: Number of processes participating in the job."""
#         return self._world_size

#     @property
#     def deterministic(self):
#         """int: Whether cudnn to select deterministic algorithms."""
#         return self._deterministic

#     @property
#     def seed(self):
#         """int: A number to set random modules."""
#         return self._seed

#     @property
#     def timestamp(self):
#         """str: Timestamp when creating experiment."""
#         return self._timestamp

#     @property
#     def hooks(self):
#         """list[:obj:`Hook`]: A list of registered hooks."""
#         return self._hooks

#     @property
#     def train_loop(self):
#         """:obj:`BaseLoop`: A loop to run training."""
#         if isinstance(self._train_loop, BaseLoop) or self._train_loop is None:
#             return self._train_loop
#         else:
#             self._train_loop = self.build_train_loop(self._train_loop)
#             return self._train_loop

#     @property
#     def val_loop(self):
#         """:obj:`BaseLoop`: A loop to run validation."""
#         if isinstance(self._val_loop, BaseLoop) or self._val_loop is None:
#             return self._val_loop
#         else:
#             self._val_loop = self.build_val_loop(self._val_loop)
#             return self._val_loop

#     @property
#     def test_loop(self):
#         """:obj:`BaseLoop`: A loop to run testing."""
#         if isinstance(self._test_loop, BaseLoop) or self._test_loop is None:
#             return self._test_loop
#         else:
#             self._test_loop = self.build_test_loop(self._test_loop)
#             return self._test_loop

#     @property
#     def train_dataloader(self):
#         """The data loader for training."""
#         return self.train_loop.dataloader

#     @property
#     def val_dataloader(self):
#         """The data loader for validation."""
#         return self.val_loop.dataloader

#     @property
#     def test_dataloader(self):
#         """The data loader for testing."""
#         return self.test_loop.dataloader

#     @property
#     def val_evaluator(self):
#         """:obj:`Evaluator`: An evaluator for validation."""
#         return self.val_loop.evaluator

#     @property
#     def test_evaluator(self):
#         """:obj:`Evaluator`: An evaluator for testing."""
#         return self.test_loop.evaluator

#     @property
#     def val_interval(self):
#         """int: Interval to run validation during training."""
#         return self.train_loop.val_interval

#     @property
#     def val_begin(self):
#         """int: The epoch/iteration to start running validation during
#         training."""
#         return self.train_loop.val_begin

#     def setup_env(self, env_cfg: Dict) -> None:
#         """Setup environment.

#         An example of ``env_cfg``::

#             env_cfg = dict(
#                 cudnn_benchmark=True,
#                 mp_cfg=dict(
#                     mp_start_method='fork',
#                     opencv_num_threads=0
#                 ),
#                 dist_cfg=dict(backend='nccl', timeout=1800),
#                 resource_limit=4096
#             )

#         Args:
#             env_cfg (dict): Config for setting environment.
#         """
#         if env_cfg.get('cudnn_benchmark'):
#             torch.backends.cudnn.benchmark = True

#         mp_cfg: dict = env_cfg.get('mp_cfg', {})
#         set_multi_processing(**mp_cfg, distributed=self.distributed)

#         # init distributed env first, since logger depends on the dist info.
#         if self.distributed and not is_distributed():
#             dist_cfg: dict = env_cfg.get('dist_cfg', {})
#             init_dist(self.launcher, **dist_cfg)

#         self._rank, self._world_size = get_dist_info()

#         timestamp = torch.tensor(time.time(), dtype=torch.float64)
#         # broadcast timestamp from 0 process to other processes
#         broadcast(timestamp)
#         self._timestamp = time.strftime('%Y%m%d_%H%M%S',
#                                         time.localtime(timestamp.item()))

#         # https://github.com/pytorch/pytorch/issues/973
#         # set resource limit
#         if platform.system() != 'Windows':
#             import resource
#             rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#             base_soft_limit = rlimit[0]
#             hard_limit = rlimit[1]
#             soft_limit = min(
#                 max(env_cfg.get('resource_limit', 4096), base_soft_limit),
#                 hard_limit)
#             resource.setrlimit(resource.RLIMIT_NOFILE,
#                                (soft_limit, hard_limit))

#     def set_randomness(self,
#                        seed,
#                        diff_rank_seed: bool = False,
#                        deterministic: bool = False) -> None:
#         """Set random seed to guarantee reproducible results.

#         Args:
#             seed (int): A number to set random modules.
#             diff_rank_seed (bool): Whether or not set different seeds according
#                 to global rank. Defaults to False.
#             deterministic (bool): Whether to set the deterministic option for
#                 CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
#                 to True and `torch.backends.cudnn.benchmark` to False.
#                 Defaults to False.
#                 See https://pytorch.org/docs/stable/notes/randomness.html for
#                 more details.
#         """
#         self._deterministic = deterministic
#         self._seed = set_random_seed(
#             seed=seed,
#             deterministic=deterministic,
#             diff_rank_seed=diff_rank_seed)

#     def build_logger(self,
#                      log_level: Union[int, str] = 'INFO',
#                      log_file: str = None,
#                      **kwargs) -> MMLogger:
#         """Build a global asscessable MMLogger.

#         Args:
#             log_level (int or str): The log level of MMLogger handlers.
#                 Defaults to 'INFO'.
#             log_file (str, optional): Path of filename to save log.
#                 Defaults to None.
#             **kwargs: Remaining parameters passed to ``MMLogger``.

#         Returns:
#             MMLogger: A MMLogger object build from ``logger``.
#         """
#         if log_file is None:
#             log_file = osp.join(self._log_dir, f'{self.timestamp}.log')

#         log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
#         log_cfg.setdefault('name', self._experiment_name)
#         # `torch.compile` in PyTorch 2.0 could close all user defined handlers
#         # unexpectedly. Using file mode 'a' can help prevent abnormal
#         # termination of the FileHandler and ensure that the log file could
#         # be continuously updated during the lifespan of the runner.
#         log_cfg.setdefault('file_mode', 'a')

#         return MMLogger.get_instance(**log_cfg)  # type: ignore

#     def build_message_hub(self,
#                           message_hub: Optional[Dict] = None) -> MessageHub:
#         """Build a global asscessable MessageHub.

#         Args:
#             message_hub (dict, optional): A dict to build MessageHub object.
#                 If not specified, default config will be used to build
#                 MessageHub object. Defaults to None.

#         Returns:
#             MessageHub: A MessageHub object build from ``message_hub``.
#         """
#         if message_hub is None:
#             message_hub = dict(name=self._experiment_name)
#         elif isinstance(message_hub, dict):
#             # ensure message_hub containing name key
#             message_hub.setdefault('name', self._experiment_name)
#         else:
#             raise TypeError(
#                 f'message_hub should be dict or None, but got {message_hub}')

#         return MessageHub.get_instance(**message_hub)

#     def build_visualizer(
#             self,
#             visualizer: Optional[Union[Visualizer,
#                                        Dict]] = None) -> Visualizer:
#         """Build a global asscessable Visualizer.

#         Args:
#             visualizer (Visualizer or dict, optional): A Visualizer object
#                 or a dict to build Visualizer object. If ``visualizer`` is a
#                 Visualizer object, just returns itself. If not specified,
#                 default config will be used to build Visualizer object.
#                 Defaults to None.

#         Returns:
#             Visualizer: A Visualizer object build from ``visualizer``.
#         """
#         if visualizer is None:
#             visualizer = dict(
#                 name=self._experiment_name,
#                 vis_backends=[dict(type='LocalVisBackend')],
#                 save_dir=self._log_dir)
#             return Visualizer.get_instance(**visualizer)

#         if isinstance(visualizer, Visualizer):
#             return visualizer

#         if isinstance(visualizer, dict):
#             # ensure visualizer containing name key
#             visualizer.setdefault('name', self._experiment_name)
#             visualizer.setdefault('save_dir', self._log_dir)
#             return VISUALIZERS.build(visualizer)
#         else:
#             raise TypeError(
#                 'visualizer should be Visualizer object, a dict or None, '
#                 f'but got {visualizer}')

#     def build_model(self, model: Union[nn.Module, Dict]) -> nn.Module:
#         """Build model.

#         If ``model`` is a dict, it will be used to build a nn.Module object.
#         Else, if ``model`` is a nn.Module object it will be returned directly.

#         An example of ``model``::

#             model = dict(type='ResNet')

#         Args:
#             model (nn.Module or dict): A ``nn.Module`` object or a dict to
#                 build nn.Module object. If ``model`` is a nn.Module object,
#                 just returns itself.

#         Note:
#             The returned model must implement ``train_step``, ``test_step``
#             if ``runner.train`` or ``runner.test`` will be called. If
#             ``runner.val`` will be called or ``val_cfg`` is configured,
#             model must implement `val_step`.

#         Returns:
#             nn.Module: Model build from ``model``.
#         """
#         if isinstance(model, nn.Module):
#             return model
#         elif isinstance(model, dict):
#             model = MODELS.build(model)
#             return model  # type: ignore
#         else:
#             raise TypeError('model should be a nn.Module object or dict, '
#                             f'but got {model}')

#     def wrap_model(
#             self, model_wrapper_cfg: Optional[Dict],
#             model: nn.Module) -> Union[DistributedDataParallel, nn.Module]:
#         """Wrap the model to :obj:`MMDistributedDataParallel` or other custom
#         distributed data-parallel module wrappers.

#         An example of ``model_wrapper_cfg``::

#             model_wrapper_cfg = dict(
#                 broadcast_buffers=False,
#                 find_unused_parameters=False
#             )

#         Args:
#             model_wrapper_cfg (dict, optional): Config to wrap model. If not
#                 specified, ``DistributedDataParallel`` will be used in
#                 distributed environment. Defaults to None.
#             model (nn.Module): Model to be wrapped.

#         Returns:
#             nn.Module or DistributedDataParallel: nn.Module or subclass of
#             ``DistributedDataParallel``.
#         """
#         if is_model_wrapper(model):
#             if model_wrapper_cfg is not None:
#                 raise TypeError(
#                     'model has been wrapped and "model_wrapper_cfg" should be '
#                     f'None, but got {model_wrapper_cfg}')

#             return model

#         # Set `export CUDA_VISIBLE_DEVICES=-1` to enable CPU training.
#         model = model.to(get_device())

#         if not self.distributed:
#             self.logger.info(
#                 'Distributed training is not used, all SyncBatchNorm (SyncBN) '
#                 'layers in the model will be automatically reverted to '
#                 'BatchNormXd layers if they are used.')
#             model = revert_sync_batchnorm(model)
#             return model  # type: ignore
#         else:
#             sync_bn = self.cfg.get('sync_bn', None)
#             if sync_bn is not None:
#                 try:
#                     model = convert_sync_batchnorm(model, sync_bn)
#                 except ValueError as e:
#                     self.logger.error('cfg.sync_bn should be "torch" or '
#                                       f'"mmcv", but got {sync_bn}')
#                     raise e
#         if model_wrapper_cfg is None:
#             find_unused_parameters = self.cfg.get('find_unused_parameters',
#                                                   False)
#             # Sets the `find_unused_parameters` parameter in
#             # torch.nn.parallel.DistributedDataParallel
#             # TODO: may use a more elegant way to get local device ID.
#             model = MMDistributedDataParallel(
#                 module=model,
#                 device_ids=[int(os.environ['LOCAL_RANK'])],
#                 broadcast_buffers=False,
#                 find_unused_parameters=find_unused_parameters)
#         else:
#             model_wrapper_cfg.setdefault('type', 'MMDistributedDataParallel')
#             model_wrapper_type = MODEL_WRAPPERS.get(
#                 model_wrapper_cfg.get('type'))  # type: ignore
#             default_args: dict = dict()
#             if issubclass(
#                     model_wrapper_type,  # type: ignore
#                     DistributedDataParallel):
#                 default_args['device_ids'] = [int(os.environ['LOCAL_RANK'])]
#             default_args['module'] = model
#             model = MODEL_WRAPPERS.build(
#                 model_wrapper_cfg, default_args=default_args)
#         return model

#     def _init_model_weights(self) -> None:
#         """Initialize the model weights if the model has
#         :meth:`init_weights`"""
#         model = self.model.module if is_model_wrapper(
#             self.model) else self.model
#         if hasattr(model, 'init_weights'):
#             model.init_weights()
#             # sync params and buffers
#             for name, params in model.state_dict().items():
#                 broadcast(params)

#     def scale_lr(self,
#                  optim_wrapper: OptimWrapper,
#                  auto_scale_lr: Optional[Dict] = None) -> None:
#         """Automatically scaling learning rate in training according to the
#         ratio of ``base_batch_size`` in ``autoscalelr_cfg`` and real batch
#         size.

#         It scales the learning rate linearly according to the
#         `paper <https://arxiv.org/abs/1706.02677>`_.

#         Note:
#             ``scale_lr`` must be called after building optimizer wrappers
#             and before building parameter schedulers.

#         Args:
#             optim_wrapper (OptimWrapper): An OptimWrapper object whose
#                 parameter groups' learning rate need to be scaled.
#             auto_scale_lr (Dict, Optional): Config to scale the learning
#                 rate automatically. It includes ``base_batch_size`` and
#                 ``enable``. ``base_batch_size`` is the batch size that the
#                 optimizer lr is based on. ``enable`` is the switch to turn on
#                 and off the feature.
#         """
#         if (auto_scale_lr is None or not auto_scale_lr.get('enable', False)):
#             return None

#         assert 'base_batch_size' in auto_scale_lr, \
#             'Lack of `base_batch_size` in `auto_scale_lr`.'
#         dataloader: Union[DataLoader, Dict] = self._train_dataloader
#         bs = dataloader.batch_size if isinstance(
#             dataloader, DataLoader) else dataloader['batch_size']
#         real_bs = self.world_size * bs
#         base_bs = auto_scale_lr['base_batch_size']
#         ratio = float(real_bs) / float(base_bs)
#         self.logger.info(f'LR is set based on batch size of {base_bs} '
#                          f'and the current batch size is {real_bs}. '
#                          f'Scaling the original LR by {ratio}.')

#         def _is_built(schedulers):
#             if isinstance(schedulers, dict):
#                 return False if 'type' in schedulers else any(
#                     _is_built(s) for s in schedulers.values())
#             if isinstance(schedulers, list):
#                 return any(_is_built(s) for s in schedulers)
#             return isinstance(schedulers, _ParamScheduler)

#         if _is_built(self.param_schedulers):
#             raise RuntimeError('`scale_lr` should be called before building '
#                                'ParamScheduler because ParamScheduler will '
#                                'store initial lr from optimizer wrappers')

#         assert isinstance(optim_wrapper, OptimWrapper), \
#             '`scale_lr should be called after building OptimWrapper'
#         wrappers = list(optim_wrapper.values()) if isinstance(
#             optim_wrapper, OptimWrapperDict) else [optim_wrapper]
#         for wrapper in wrappers:
#             for group in wrapper.optimizer.param_groups:
#                 group['lr'] = group['lr'] * ratio

#     def build_optim_wrapper(
#         self, optim_wrapper: Union[Optimizer, OptimWrapper, Dict]
#     ) -> Union[OptimWrapper, OptimWrapperDict]:
#         """Build optimizer wrapper.

#         If ``optim_wrapper`` is a config dict for only one optimizer,
#         the keys must contain ``optimizer``, and ``type`` is optional.
#         It will build a :obj:`OptimWrapper` by default.

#         If ``optim_wrapper`` is a config dict for multiple optimizers, i.e.,
#         it has multiple keys and each key is for an optimizer wrapper. The
#         constructor must be specified since
#         :obj:`DefaultOptimizerConstructor` cannot handle the building of
#         training with multiple optimizers.

#         If ``optim_wrapper`` is a dict of pre-built optimizer wrappers, i.e.,
#         each value of ``optim_wrapper`` represents an ``OptimWrapper``
#         instance. ``build_optim_wrapper`` will directly build the
#         :obj:`OptimWrapperDict` instance from ``optim_wrapper``.

#         Args:
#             optim_wrapper (OptimWrapper or dict): An OptimWrapper object or a
#                 dict to build OptimWrapper objects. If ``optim_wrapper`` is an
#                 OptimWrapper, just return an ``OptimizeWrapper`` instance.

#         Note:
#             For single optimizer training, if `optim_wrapper` is a config
#             dict, `type` is optional(defaults to :obj:`OptimWrapper`) and it
#             must contain `optimizer` to build the corresponding optimizer.

#         Examples:
#             >>> # build an optimizer
#             >>> optim_wrapper_cfg = dict(type='OptimWrapper', optimizer=dict(
#             ...     type='SGD', lr=0.01))
#             >>> # optim_wrapper_cfg = dict(optimizer=dict(type='SGD', lr=0.01))
#             >>> # is also valid.
#             >>> optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
#             >>> optim_wrapper
#             Type: OptimWrapper
#             accumulative_counts: 1
#             optimizer:
#             SGD (
#             Parameter Group 0
#                 dampening: 0
#                 lr: 0.01
#                 momentum: 0
#                 nesterov: False
#                 weight_decay: 0
#             )
#             >>> # build optimizer without `type`
#             >>> optim_wrapper_cfg = dict(optimizer=dict(type='SGD', lr=0.01))
#             >>> optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
#             >>> optim_wrapper
#             Type: OptimWrapper
#             accumulative_counts: 1
#             optimizer:
#             SGD (
#             Parameter Group 0
#                 dampening: 0
#                 lr: 0.01
#                 maximize: False
#                 momentum: 0
#                 nesterov: False
#                 weight_decay: 0
#             )
#             >>> # build multiple optimizers
#             >>> optim_wrapper_cfg = dict(
#             ...    generator=dict(type='OptimWrapper', optimizer=dict(
#             ...        type='SGD', lr=0.01)),
#             ...    discriminator=dict(type='OptimWrapper', optimizer=dict(
#             ...        type='Adam', lr=0.001))
#             ...    # need to customize a multiple optimizer constructor
#             ...    constructor='CustomMultiOptimizerConstructor',
#             ...)
#             >>> optim_wrapper = runner.optim_wrapper(optim_wrapper_cfg)
#             >>> optim_wrapper
#             name: generator
#             Type: OptimWrapper
#             accumulative_counts: 1
#             optimizer:
#             SGD (
#             Parameter Group 0
#                 dampening: 0
#                 lr: 0.1
#                 momentum: 0
#                 nesterov: False
#                 weight_decay: 0
#             )
#             name: discriminator
#             Type: OptimWrapper
#             accumulative_counts: 1
#             optimizer:
#             'discriminator': Adam (
#             Parameter Group 0
#                 dampening: 0
#                 lr: 0.02
#                 momentum: 0
#                 nesterov: False
#                 weight_decay: 0
#             )

#         Important:
#             If you need to build multiple optimizers, you should implement a
#             MultiOptimWrapperConstructor which gets parameters passed to
#             corresponding optimizers and compose the ``OptimWrapperDict``.
#             More details about how to customize OptimizerConstructor can be
#             found at `optimizer-docs`_.

#         Returns:
#             OptimWrapper: Optimizer wrapper build from ``optimizer_cfg``.
#         """  # noqa: E501
#         if isinstance(optim_wrapper, OptimWrapper):
#             return optim_wrapper
#         if isinstance(optim_wrapper, (dict, ConfigDict, Config)):
#             # optimizer must be defined for single optimizer training.
#             optimizer = optim_wrapper.get('optimizer', None)

#             # If optimizer is a built `Optimizer` instance, the optimizer
#             # wrapper should be built by `OPTIM_WRAPPERS` registry.
#             if isinstance(optimizer, Optimizer):
#                 optim_wrapper.setdefault('type', 'OptimWrapper')
#                 return OPTIM_WRAPPERS.build(optim_wrapper)  # type: ignore

#             # If `optimizer` is not None or `constructor` is defined, it means,
#             # optimizer wrapper will be built by optimizer wrapper
#             # constructor. Therefore, `build_optim_wrapper` should be called.
#             if optimizer is not None or 'constructor' in optim_wrapper:
#                 return build_optim_wrapper(self.model, optim_wrapper)
#             else:
#                 # if `optimizer` is not defined, it should be the case of
#                 # training with multiple optimizers. If `constructor` is not
#                 # defined either, each value of `optim_wrapper` must be an
#                 # `OptimWrapper` instance since `DefaultOptimizerConstructor`
#                 # will not handle the case of training with multiple
#                 # optimizers. `build_optim_wrapper` will directly build the
#                 # `OptimWrapperDict` instance from `optim_wrapper.`
#                 optim_wrappers = OrderedDict()
#                 for name, optim in optim_wrapper.items():
#                     if not isinstance(optim, OptimWrapper):
#                         raise ValueError(
#                             'each item mush be an optimizer object when '
#                             '"type" and "constructor" are not in '
#                             f'optimizer, but got {name}={optim}')
#                     optim_wrappers[name] = optim
#                 return OptimWrapperDict(**optim_wrappers)
#         else:
#             raise TypeError('optimizer wrapper should be an OptimWrapper '
#                             f'object or dict, but got {optim_wrapper}')

#     def _build_param_scheduler(
#             self, scheduler: Union[_ParamScheduler, Dict, List],
#             optim_wrapper: OptimWrapper) -> List[_ParamScheduler]:
#         """Build parameter schedulers for a single optimizer.

#         Args:
#             scheduler (_ParamScheduler or dict or list): A Param Scheduler
#                 object or a dict or list of dict to build parameter schedulers.
#             optim_wrapper (OptimWrapper): An optimizer wrapper object is
#                 passed to construct ParamScheduler object.

#         Returns:
#             list[_ParamScheduler]: List of parameter schedulers build from
#             ``scheduler``.

#         Note:
#             If the train loop is built, when building parameter schedulers,
#             it supports setting the max epochs/iters as the default ``end``
#             of schedulers, and supports converting epoch-based schedulers
#             to iter-based according to the ``convert_to_iter_based`` key.
#         """
#         if not isinstance(scheduler, Sequence):
#             schedulers = [scheduler]
#         else:
#             schedulers = scheduler

#         param_schedulers = []
#         for scheduler in schedulers:
#             if isinstance(scheduler, _ParamScheduler):
#                 param_schedulers.append(scheduler)
#             elif isinstance(scheduler, dict):
#                 _scheduler = copy.deepcopy(scheduler)

#                 # Set default end
#                 if isinstance(self._train_loop, BaseLoop):
#                     default_end = self.max_epochs if _scheduler.get(
#                         'by_epoch', True) else self.max_iters
#                     _scheduler.setdefault('end', default_end)
#                     self.logger.debug(
#                         f'The `end` of {_scheduler["type"]} is not set. '
#                         'Use the max epochs/iters of train loop as default.')

#                 param_schedulers.append(
#                     PARAM_SCHEDULERS.build(
#                         _scheduler,
#                         default_args=dict(
#                             optimizer=optim_wrapper,
#                             epoch_length=len(self.train_dataloader))))
#             else:
#                 raise TypeError(
#                     'scheduler should be a _ParamScheduler object or dict, '
#                     f'but got {scheduler}')
#         return param_schedulers

#     def build_param_scheduler(
#             self, scheduler: Union[_ParamScheduler, Dict,
#                                    List]) -> ParamSchedulerType:
#         """Build parameter schedulers.

#         ``build_param_scheduler`` should be called after
#         ``build_optim_wrapper`` because the building logic will change
#         according to the number of optimizers built by the runner.
#         The cases are as below:

#         - Single optimizer: When only one optimizer is built and used in the
#           runner, ``build_param_scheduler`` will return a list of
#           parameter schedulers.
#         - Multiple optimizers: When two or more optimizers are built and used
#           in runner, ``build_param_scheduler`` will return a dict containing
#           the same keys with multiple optimizers and each value is a list of
#           parameter schedulers. Note that, if you want different optimizers to
#           use different parameter schedulers to update optimizer's
#           hyper-parameters, the input parameter ``scheduler`` also needs to be
#           a dict and its key are consistent with multiple optimizers.
#           Otherwise, the same parameter schedulers will be used to update
#           optimizer's hyper-parameters.

#         Args:
#             scheduler (_ParamScheduler or dict or list): A Param Scheduler
#                 object or a dict or list of dict to build parameter schedulers.

#         Examples:
#             >>> # build one scheduler
#             >>> optim_cfg = dict(dict(type='SGD', lr=0.01))
#             >>> runner.optim_wrapper = runner.build_optim_wrapper(
#             >>>     optim_cfg)
#             >>> scheduler_cfg = dict(type='MultiStepLR', milestones=[1, 2])
#             >>> schedulers = runner.build_param_scheduler(scheduler_cfg)
#             >>> schedulers
#             [<mmengine.optim.scheduler.lr_scheduler.MultiStepLR at 0x7f70f6966290>]  # noqa: E501

#             >>> # build multiple schedulers
#             >>> scheduler_cfg = [
#             ...    dict(type='MultiStepLR', milestones=[1, 2]),
#             ...    dict(type='StepLR', step_size=1)
#             ... ]
#             >>> schedulers = runner.build_param_scheduler(scheduler_cfg)
#             >>> schedulers
#             [<mmengine.optim.scheduler.lr_scheduler.MultiStepLR at 0x7f70f60dd3d0>,  # noqa: E501
#             <mmengine.optim.scheduler.lr_scheduler.StepLR at 0x7f70f6eb6150>]

#         Above examples only provide the case of one optimizer and one scheduler
#         or multiple schedulers. If you want to know how to set parameter
#         scheduler when using multiple optimizers, you can find more examples
#         `optimizer-docs`_.

#         Returns:
#             list[_ParamScheduler] or dict[str, list[_ParamScheduler]]: List of
#             parameter schedulers or a dictionary contains list of parameter
#             schedulers build from ``scheduler``.

#         .. _optimizer-docs:
#            https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html
#         """
#         param_schedulers: ParamSchedulerType
#         if not isinstance(self.optim_wrapper, OptimWrapperDict):
#             # Since `OptimWrapperDict` inherits from `OptimWrapper`,
#             # `isinstance(self.optim_wrapper, OptimWrapper)` cannot tell
#             # whether `self.optim_wrapper` is an `OptimizerWrapper` or
#             # `OptimWrapperDict` instance. Therefore, here we simply check
#             # self.optim_wrapper is not an `OptimWrapperDict` instance and
#             # then assert it is an OptimWrapper instance.
#             assert isinstance(self.optim_wrapper, OptimWrapper), (
#                 '`build_optimizer` should be called before'
#                 '`build_param_scheduler` because the latter depends '
#                 'on the former')
#             param_schedulers = self._build_param_scheduler(
#                 scheduler, self.optim_wrapper)  # type: ignore
#             return param_schedulers
#         else:
#             param_schedulers = dict()
#             for name, optimizer in self.optim_wrapper.items():
#                 if isinstance(scheduler, dict) and 'type' not in scheduler:
#                     # scheduler is a dict and each item is a ParamScheduler
#                     # object or a config to build ParamScheduler objects
#                     param_schedulers[name] = self._build_param_scheduler(
#                         scheduler[name], optimizer)
#                 else:
#                     param_schedulers[name] = self._build_param_scheduler(
#                         scheduler, optimizer)

#             return param_schedulers

#     def build_evaluator(self, evaluator: Union[Dict, List,
#                                                Evaluator]) -> Evaluator:
#         """Build evaluator.

#         Examples of ``evaluator``::

#             # evaluator could be a built Evaluator instance
#             evaluator = Evaluator(metrics=[ToyMetric()])

#             # evaluator can also be a list of dict
#             evaluator = [
#                 dict(type='ToyMetric1'),
#                 dict(type='ToyEvaluator2')
#             ]

#             # evaluator can also be a list of built metric
#             evaluator = [ToyMetric1(), ToyMetric2()]

#             # evaluator can also be a dict with key metrics
#             evaluator = dict(metrics=ToyMetric())
#             # metric is a list
#             evaluator = dict(metrics=[ToyMetric()])

#         Args:
#             evaluator (Evaluator or dict or list): An Evaluator object or a
#                 config dict or list of config dict used to build an Evaluator.

#         Returns:
#             Evaluator: Evaluator build from ``evaluator``.
#         """
#         if isinstance(evaluator, Evaluator):
#             return evaluator
#         elif isinstance(evaluator, dict):
#             # if `metrics` in dict keys, it means to build customized evalutor
#             if 'metrics' in evaluator:
#                 evaluator.setdefault('type', 'Evaluator')
#                 return EVALUATOR.build(evaluator)
#             # otherwise, default evalutor will be built
#             else:
#                 return Evaluator(evaluator)  # type: ignore
#         elif isinstance(evaluator, list):
#             # use the default `Evaluator`
#             return Evaluator(evaluator)  # type: ignore
#         else:
#             raise TypeError(
#                 'evaluator should be one of dict, list of dict, and Evaluator'
#                 f', but got {evaluator}')

#     @staticmethod
#     def build_dataloader(dataloader: Union[DataLoader, Dict],
#                          seed: Optional[int] = None,
#                          diff_rank_seed: bool = False) -> DataLoader:
#         """Build dataloader.
# build_val_loopv
#         The method builds three components:

#         - Dataset
#         - Sampler
#         - Dataloader

#         An example of ``dataloader``::

#             dataloader = dict(
#                 dataset=dict(type='ToyDataset'),
#                 sampler=dict(type='DefaultSampler', shuffle=True),
#                 batch_size=1,
#                 num_workers=9
#             )

#         Args:
#             dataloader (DataLoader or dict): A Dataloader object or a dict to
#                 build Dataloader object. If ``dataloader`` is a Dataloader
#                 object, just returns itself.
#             seed (int, optional): Random seed. Defaults to None.
#             diff_rank_seed (bool): Whether or not set different seeds to
#                 different ranks. If True, the seed passed to sampler is set
#                 to None, in order to synchronize the seeds used in samplers
#                 across different ranks.


#         Returns:
#             Dataloader: DataLoader build from ``dataloader_cfg``.
#         """
#         if isinstance(dataloader, DataLoader):
#             return dataloader

#         dataloader_cfg = copy.deepcopy(dataloader)

#         # build dataset
#         dataset_cfg = dataloader_cfg.pop('dataset')
#         if isinstance(dataset_cfg, dict):
#             dataset = DATASETS.build(dataset_cfg)
#             if hasattr(dataset, 'full_init'):
#                 dataset.full_init()
#         else:
#             # fallback to raise error in dataloader
#             # if `dataset_cfg` is not a valid type
#             dataset = dataset_cfg

#         num_batch_per_epoch = dataloader_cfg.pop('num_batch_per_epoch', None)
#         if num_batch_per_epoch is not None:
#             world_size = get_world_size()
#             num_samples = (
#                 num_batch_per_epoch * _get_batch_size(dataloader_cfg) *
#                 world_size)
#             dataset = _SlicedDataset(dataset, num_samples)

#         # build sampler
#         sampler_cfg = dataloader_cfg.pop('sampler')
#         if isinstance(sampler_cfg, dict):
#             sampler_seed = None if diff_rank_seed else seed
#             sampler = DATA_SAMPLERS.build(
#                 sampler_cfg,
#                 default_args=dict(dataset=dataset, seed=sampler_seed))
#         else:
#             # fallback to raise error in dataloader
#             # if `sampler_cfg` is not a valid type
#             sampler = sampler_cfg

#         # build batch sampler
#         batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
#         if batch_sampler_cfg is None:
#             batch_sampler = None
#         elif isinstance(batch_sampler_cfg, dict):
#             batch_sampler = DATA_SAMPLERS.build(
#                 batch_sampler_cfg,
#                 default_args=dict(
#                     sampler=sampler,
#                     batch_size=dataloader_cfg.pop('batch_size')))
#         else:
#             # fallback to raise error in dataloader
#             # if `batch_sampler_cfg` is not a valid type
#             batch_sampler = batch_sampler_cfg

#         # build dataloader
#         init_fn: Optional[partial]

#         if 'worker_init_fn' in dataloader_cfg:
#             worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
#             worker_init_fn_type = worker_init_fn_cfg.pop('type')
#             if isinstance(worker_init_fn_type, str):
#                 worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
#             elif callable(worker_init_fn_type):
#                 worker_init_fn = worker_init_fn_type
#             else:
#                 raise TypeError(
#                     'type of worker_init_fn should be string or callable '
#                     f'object, but got {type(worker_init_fn_type)}')
#             assert callable(worker_init_fn)
#             init_fn = partial(worker_init_fn,
#                               **worker_init_fn_cfg)  # type: ignore
#         else:
#             if seed is not None:
#                 disable_subprocess_warning = dataloader_cfg.pop(
#                     'disable_subprocess_warning', False)
#                 assert isinstance(disable_subprocess_warning, bool), (
#                     'disable_subprocess_warning should be a bool, but got '
#                     f'{type(disable_subprocess_warning)}')
#                 init_fn = partial(
#                     default_worker_init_fn,
#                     num_workers=dataloader_cfg.get('num_workers'),
#                     rank=get_rank(),
#                     seed=seed,
#                     disable_subprocess_warning=disable_subprocess_warning)
#             else:
#                 init_fn = None

#         # `persistent_workers` requires pytorch version >= 1.7
#         if ('persistent_workers' in dataloader_cfg
#                 and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
#             print_log(
#                 '`persistent_workers` is only available when '
#                 'pytorch version >= 1.7',
#                 logger='current',
#                 level=logging.WARNING)
#             dataloader_cfg.pop('persistent_workers')

#         # The default behavior of `collat_fn` in dataloader is to
#         # merge a list of samples to form a mini-batch of Tensor(s).
#         # However, in mmengine, if `collate_fn` is not defined in
#         # dataloader_cfg, `pseudo_collate` will only convert the list of
#         # samples into a dict without stacking the batch tensor.
#         collate_fn_cfg = dataloader_cfg.pop('collate_fn',
#                                             dict(type='pseudo_collate'))
#         if isinstance(collate_fn_cfg, dict):
#             collate_fn_type = collate_fn_cfg.pop('type')
#             if isinstance(collate_fn_type, str):
#                 collate_fn = FUNCTIONS.get(collate_fn_type)
#             else:
#                 collate_fn = collate_fn_type
#             collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
#         elif callable(collate_fn_cfg):
#             collate_fn = collate_fn_cfg
#         else:
#             raise TypeError(
#                 'collate_fn should be a dict or callable object, but got '
#                 f'{collate_fn_cfg}')
#         data_loader = DataLoader(
#             dataset=dataset,
#             sampler=sampler if batch_sampler is None else None,
#             batch_sampler=batch_sampler,
#             collate_fn=collate_fn,
#             worker_init_fn=init_fn,
#             **dataloader_cfg)
#         return data_loader

#     def build_train_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
#         """Build training loop.

#         Examples of ``loop``::

#             # `EpochBasedTrainLoop` will be used
#             loop = dict(by_epoch=True, max_epochs=3)

#             # `IterBasedTrainLoop` will be used
#             loop = dict(by_epoch=False, max_epochs=3)

#             # custom training loop
#             loop = dict(type='CustomTrainLoop', max_epochs=3)

#         Args:
#             loop (BaseLoop or dict): A training loop or a dict to build
#                 training loop. If ``loop`` is a training loop object, just
#                 returns itself.

#         Returns:
#             :obj:`BaseLoop`: Training loop object build from ``loop``.
#         """
#         if isinstance(loop, BaseLoop):
#             return loop
#         elif not isinstance(loop, dict):
#             raise TypeError(
#                 f'train_loop should be a Loop object or dict, but got {loop}')

#         loop_cfg = copy.deepcopy(loop)

#         if 'type' in loop_cfg and 'by_epoch' in loop_cfg:
#             raise RuntimeError(
#                 'Only one of `type` or `by_epoch` can exist in `loop_cfg`.')

#         if 'type' in loop_cfg:
#             loop = LOOPS.build(
#                 loop_cfg,
#                 default_args=dict(
#                     runner=self, dataloader=self._train_dataloader))
#         else:
#             by_epoch = loop_cfg.pop('by_epoch')
#             if by_epoch:
#                 loop = EpochBasedTrainLoop(
#                     **loop_cfg, runner=self, dataloader=self._train_dataloader)
#             else:
#                 loop = IterBasedTrainLoop(
#                     **loop_cfg, runner=self, dataloader=self._train_dataloader)
#         return loop  # type: ignore

#     def build_val_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
#         """Build validation loop.

#         Examples of ``loop``:

#             # `ValLoop` will be used
#             loop = dict()

#             # custom validation loop
#             loop = dict(type='CustomValLoop')

#         Args:
#             loop (BaseLoop or dict): A validation loop or a dict to build
#                 validation loop. If ``loop`` is a validation loop object, just
#                 returns itself.

#         Returns:
#             :obj:`BaseLoop`: Validation loop object build from ``loop``.
#         """
#         if isinstance(loop, BaseLoop):
#             return loop
#         elif not isinstance(loop, dict):
#             raise TypeError(
#                 f'val_loop should be a Loop object or dict, but got {loop}')

#         loop_cfg = copy.deepcopy(loop)

#         if 'type' in loop_cfg:
#             loop = LOOPS.build(
#                 loop_cfg,
#                 default_args=dict(
#                     runner=self,
#                     dataloader=self._val_dataloader,
#                     evaluator=self._val_evaluator))
#         else:
#             loop = ValLoop(    #  ValLoop -> EpochBasedValLoop
#                 **loop_cfg,
#                 runner=self,
#                 dataloader=self._val_dataloader,
#                 evaluator=self._val_evaluator)  # type: ignore

#         return loop  # type: ignore

#     def build_test_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
#         """Build test loop.

#         Examples of ``loop``::

#             # `TestLoop` will be used
#             loop = dict()

#             # custom test loop
#             loop = dict(type='CustomTestLoop')

#         Args:
#             loop (BaseLoop or dict): A test loop or a dict to build test loop.
#                 If ``loop`` is a test loop object, just returns itself.

#         Returns:
#             :obj:`BaseLoop`: Test loop object build from ``loop_cfg``.
#         """
#         if isinstance(loop, BaseLoop):
#             return loop
#         elif not isinstance(loop, dict):
#             raise TypeError(
#                 f'test_loop should be a Loop object or dict, but got {loop}')

#         loop_cfg = copy.deepcopy(loop)  # type: ignore

#         if 'type' in loop_cfg:
#             loop = LOOPS.build(
#                 loop_cfg,
#                 default_args=dict(
#                     runner=self,
#                     dataloader=self._test_dataloader,
#                     evaluator=self._test_evaluator))
#         else:
#             loop = TestLoop(
#                 **loop_cfg,
#                 runner=self,
#                 dataloader=self._test_dataloader,
#                 evaluator=self._test_evaluator)  # type: ignore

#         return loop  # type: ignore

#     def build_log_processor(
#             self, log_processor: Union[LogProcessor, Dict]) -> LogProcessor:
#         """Build test log_processor.

#         Examples of ``log_processor``:

#             # `LogProcessor` will be used
#             log_processor = dict()

#             # custom log_processor
#             log_processor = dict(type='CustomLogProcessor')

#         Args:
#             log_processor (LogProcessor or dict): A log processor or a dict
#             to build log processor. If ``log_processor`` is a log processor
#             object, just returns itself.

#         Returns:
#             :obj:`LogProcessor`: Log processor object build from
#             ``log_processor_cfg``.
#         """
#         if isinstance(log_processor, LogProcessor):
#             return log_processor
#         elif not isinstance(log_processor, dict):
#             raise TypeError(
#                 'log processor should be a LogProcessor object or dict, but'
#                 f'got {log_processor}')

#         log_processor_cfg = copy.deepcopy(log_processor)  # type: ignore

#         if 'type' in log_processor_cfg:
#             log_processor = LOG_PROCESSORS.build(log_processor_cfg)
#         else:
#             log_processor = LogProcessor(**log_processor_cfg)  # type: ignore

#         return log_processor  # type: ignore

#     def get_hooks_info(self) -> str:
#         # Get hooks info in each stage
#         stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
#         for hook in self.hooks:
#             try:
#                 priority = Priority(hook.priority).name  # type: ignore
#             except ValueError:
#                 priority = hook.priority  # type: ignore
#             classname = hook.__class__.__name__
#             hook_info = f'({priority:<12}) {classname:<35}'
#             for trigger_stage in hook.get_triggered_stages():
#                 stage_hook_map[trigger_stage].append(hook_info)

#         stage_hook_infos = []
#         for stage in Hook.stages:
#             hook_infos = stage_hook_map[stage]
#             if len(hook_infos) > 0:
#                 info = f'{stage}:\n'
#                 info += '\n'.join(hook_infos)
#                 info += '\n -------------------- '
#                 stage_hook_infos.append(info)
#         return '\n'.join(stage_hook_infos)

#     def load_or_resume(self) -> None:
#         """load or resume checkpoint."""
#         if self._has_loaded:
#             return None

#         # decide to load from checkpoint or resume from checkpoint
#         resume_from = None
#         if self._resume and self._load_from is None:
#             # auto resume from the latest checkpoint
#             resume_from = find_latest_checkpoint(self.work_dir)
#             self.logger.info(
#                 f'Auto resumed from the latest checkpoint {resume_from}.')
#         elif self._resume and self._load_from is not None:
#             # resume from the specified checkpoint
#             resume_from = self._load_from

#         if resume_from is not None:
#             self.resume(resume_from)
#             self._has_loaded = True
#         elif self._load_from is not None:
#             self.load_checkpoint(self._load_from)
#             self._has_loaded = True

#     def train(self) -> nn.Module:
#         """Launch training.

#         Returns:
#             nn.Module: The model after training.
#         """
#         if is_model_wrapper(self.model):
#             ori_model = self.model.module
#         else:
#             ori_model = self.model
#         assert hasattr(ori_model, 'train_step'), (
#             'If you want to train your model, please make sure your model '
#             'has implemented `train_step`.')

#         if self._val_loop is not None:
#             assert hasattr(ori_model, 'val_step'), (
#                 'If you want to validate your model, please make sure your '
#                 'model has implemented `val_step`.')

#         if self._train_loop is None:
#             raise RuntimeError(
#                 '`self._train_loop` should not be None when calling train '
#                 'method. Please provide `train_dataloader`, `train_cfg`, '
#                 '`optimizer` and `param_scheduler` arguments when '
#                 'initializing runner.')

#         self._train_loop = self.build_train_loop(
#             self._train_loop)  # type: ignore

#         # `build_optimizer` should be called before `build_param_scheduler`
#         #  because the latter depends on the former
#         self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
#         # Automatically scaling lr by linear scaling rule
#         self.scale_lr(self.optim_wrapper, self.auto_scale_lr)

#         if self.param_schedulers is not None:
#             self.param_schedulers = self.build_param_scheduler(  # type: ignore
#                 self.param_schedulers)  # type: ignore

#         if self._val_loop is not None:
#             self._val_loop = self.build_val_loop(
#                 self._val_loop)  # type: ignore
#         # TODO: add a contextmanager to avoid calling `before_run` many times
#         self.call_hook('before_run')

#         # initialize the model weights
#         self._init_model_weights()

#         # try to enable activation_checkpointing feature
#         modules = self.cfg.get('activation_checkpointing', None)
#         if modules is not None:
#             self.logger.info(f'Enabling the "activation_checkpointing" feature'
#                              f' for sub-modules: {modules}')
#             turn_on_activation_checkpointing(ori_model, modules)

#         # try to enable efficient_conv_bn_eval feature
#         modules = self.cfg.get('efficient_conv_bn_eval', None)
#         if modules is not None:
#             self.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
#                              f' for sub-modules: {modules}')
#             turn_on_efficient_conv_bn_eval(ori_model, modules)

#         # make sure checkpoint-related hooks are triggered after `before_run`
#         self.load_or_resume()

#         # Initiate inner count of `optim_wrapper`.
#         self.optim_wrapper.initialize_count_status(
#             self.model,
#             self._train_loop.iter,  # type: ignore
#             self._train_loop.max_iters)  # type: ignore

#         # Maybe compile the model according to options in self.cfg.compile
#         # This must be called **AFTER** model has been wrapped.
#         self._maybe_compile('train_step')

#         model = self.train_loop.run()  # type: ignore
#         self.call_hook('after_run')
#         return model

#     def val(self) -> dict:
#         """Launch validation.

#         Returns:
#             dict: A dict of metrics on validation set.
#         """
#         if self._val_loop is None:
#             raise RuntimeError(
#                 '`self._val_loop` should not be None when calling val method.'
#                 'Please provide `val_dataloader`, `val_cfg` and '
#                 '`val_evaluator` arguments when initializing runner.')

#         self._val_loop = self.build_val_loop(self._val_loop)  # type: ignore

#         self.call_hook('before_run')

#         # make sure checkpoint-related hooks are triggered after `before_run`
#         self.load_or_resume()

#         metrics = self.val_loop.run()  # type: ignore
#         # 검증 루프 실행 및 로그 기록
#         val_results = self.val_loop.run()
#         val_loss = val_results['loss']  # 검증 손실 가져오기
#         self.logger.info(f'Validation Loss: {val_loss}')  # 로그에 손실 기록
#         self.call_hook('after_run')
#         return metrics

#     def test(self) -> dict:
#         """Launch test.

#         Returns:
#             dict: A dict of metrics on testing set.
#         """
#         if self._test_loop is None:
#             raise RuntimeError(
#                 '`self._test_loop` should not be None when calling test '
#                 'method. Please provide `test_dataloader`, `test_cfg` and '
#                 '`test_evaluator` arguments when initializing runner.')

#         self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore

#         self.call_hook('before_run')

#         # make sure checkpoint-related hooks are triggered after `before_run`
#         self.load_or_resume()

#         metrics = self.test_loop.run()  # type: ignore
#         self.call_hook('after_run')
#         return metrics

#     def call_hook(self, fn_name: str, **kwargs) -> None:
#         """Call all hooks.

#         Args:
#             fn_name (str): The function name in each hook to be called, such as
#                 "before_train_epoch".
#             **kwargs: Keyword arguments passed to hook.
#         """
#         for hook in self._hooks:
#             # support adding additional custom hook methods
#             if hasattr(hook, fn_name):
#                 try:
#                     getattr(hook, fn_name)(self, **kwargs)
#                 except TypeError as e:
#                     raise TypeError(f'{e} in {hook}') from None

#     def register_hook(
#             self,
#             hook: Union[Hook, Dict],
#             priority: Optional[Union[str, int, Priority]] = None) -> None:
#         """Register a hook into the hook list.

#         The hook will be inserted into a priority queue, with the specified
#         priority (See :class:`Priority` for details of priorities).
#         For hooks with the same priority, they will be triggered in the same
#         order as they are registered.

#         Priority of hook will be decided with the following priority:

#         - ``priority`` argument. If ``priority`` is given, it will be priority
#           of hook.
#         - If ``hook`` argument is a dict and ``priority`` in it, the priority
#           will be the value of ``hook['priority']``.
#         - If ``hook`` argument is a dict but ``priority`` not in it or ``hook``
#           is an instance of ``hook``, the priority will be ``hook.priority``.

#         Args:
#             hook (:obj:`Hook` or dict): The hook to be registered.
#             priority (int or str or :obj:`Priority`, optional): Hook priority.
#                 Lower value means higher priority.
#         """
#         if not isinstance(hook, (Hook, dict)):
#             raise TypeError(
#                 f'hook should be an instance of Hook or dict, but got {hook}')

#         _priority = None
#         if isinstance(hook, dict):
#             if 'priority' in hook:
#                 _priority = hook.pop('priority')

#             hook_obj = HOOKS.build(hook)
#         else:
#             hook_obj = hook

#         if priority is not None:
#             hook_obj.priority = priority
#         elif _priority is not None:
#             hook_obj.priority = _priority

#         inserted = False
#         for i in range(len(self._hooks) - 1, -1, -1):
#             if get_priority(hook_obj.priority) >= get_priority(
#                     self._hooks[i].priority):
#                 self._hooks.insert(i + 1, hook_obj)
#                 inserted = True
#                 break
#         if not inserted:
#             self._hooks.insert(0, hook_obj)

#     def register_default_hooks(
#             self,
#             hooks: Optional[Dict[str, Union[Hook, Dict]]] = None) -> None:
#         """Register default hooks into hook list.

#         ``hooks`` will be registered into runner to execute some default
#         actions like updating model parameters or saving checkpoints.

#         Default hooks and their priorities:

#         +----------------------+-------------------------+
#         | Hooks                | Priority                |
#         +======================+=========================+
#         | RuntimeInfoHook      | VERY_HIGH (10)          |
#         +----------------------+-------------------------+
#         | IterTimerHook        | NORMAL (50)             |
#         +----------------------+-------------------------+
#         | DistSamplerSeedHook  | NORMAL (50)             |
#         +----------------------+-------------------------+
#         | LoggerHook           | BELOW_NORMAL (60)       |
#         +----------------------+-------------------------+
#         | ParamSchedulerHook   | LOW (70)                |
#         +----------------------+-------------------------+
#         | CheckpointHook       | VERY_LOW (90)           |
#         +----------------------+-------------------------+

#         If ``hooks`` is None, above hooks will be registered by
#         default::

#             default_hooks = dict(
#                 runtime_info=dict(type='RuntimeInfoHook'),
#                 timer=dict(type='IterTimerHook'),
#                 sampler_seed=dict(type='DistSamplerSeedHook'),
#                 logger=dict(type='LoggerHook'),
#                 param_scheduler=dict(type='ParamSchedulerHook'),
#                 checkpoint=dict(type='CheckpointHook', interval=1),
#             )

#         If not None, ``hooks`` will be merged into ``default_hooks``.
#         If there are None value in default_hooks, the corresponding item will
#         be popped from ``default_hooks``::

#             hooks = dict(timer=None)

#         The final registered default hooks will be :obj:`RuntimeInfoHook`,
#         :obj:`DistSamplerSeedHook`, :obj:`LoggerHook`,
#         :obj:`ParamSchedulerHook` and :obj:`CheckpointHook`.

#         Args:
#             hooks (dict[str, Hook or dict], optional): Default hooks or configs
#                 to be registered.
#         """
#         default_hooks: dict = dict(
#             runtime_info=dict(type='RuntimeInfoHook'),
#             timer=dict(type='IterTimerHook'),
#             sampler_seed=dict(type='DistSamplerSeedHook'),
#             logger=dict(type='LoggerHook'),
#             param_scheduler=dict(type='ParamSchedulerHook'),
#             checkpoint=dict(type='CheckpointHook', interval=1),
#         )
#         if hooks is not None:
#             for name, hook in hooks.items():
#                 if name in default_hooks and hook is None:
#                     # remove hook from _default_hooks
#                     default_hooks.pop(name)
#                 else:
#                     assert hook is not None
#                     default_hooks[name] = hook

#         for hook in default_hooks.values():
#             self.register_hook(hook)

#     def register_custom_hooks(self, hooks: List[Union[Hook, Dict]]) -> None:
#         """Register custom hooks into hook list.

#         Args:
#             hooks (list[Hook | dict]): List of hooks or configs to be
#                 registered.
#         """
#         for hook in hooks:
#             self.register_hook(hook)

#     def register_hooks(
#             self,
#             default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
#             custom_hooks: Optional[List[Union[Hook, Dict]]] = None) -> None:
#         """Register default hooks and custom hooks into hook list.

#         Args:
#             default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks
#                 to execute default actions like updating model parameters and
#                 saving checkpoints.  Defaults to None.
#             custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
#                 custom actions like visualizing images processed by pipeline.
#                 Defaults to None.
#         """
#         self.register_default_hooks(default_hooks)

#         if custom_hooks is not None:
#             self.register_custom_hooks(custom_hooks)

#     def resume(self,
#                filename: str,
#                resume_optimizer: bool = True,
#                resume_param_scheduler: bool = True,
#                map_location: Union[str, Callable] = 'default') -> None:
#         """Resume model from checkpoint.

#         Args:
#             filename (str): Accept local filepath, URL, ``torchvision://xxx``,
#                 ``open-mmlab://xxx``.
#             resume_optimizer (bool): Whether to resume optimizer state.
#                 Defaults to True.
#             resume_param_scheduler (bool): Whether to resume param scheduler
#                 state. Defaults to True.
#             map_location (str or callable):A string or a callable function to
#                 specifying how to remap storage locations.
#                 Defaults to 'default'.
#         """
#         if map_location == 'default':
#             device = get_device()
#             checkpoint = self.load_checkpoint(filename, map_location=device)
#         else:
#             checkpoint = self.load_checkpoint(
#                 filename, map_location=map_location)

#         self.train_loop._epoch = checkpoint['meta']['epoch']
#         self.train_loop._iter = checkpoint['meta']['iter']

#         # check whether the number of GPU used for current experiment
#         # is consistent with resuming from checkpoint
#         if 'config' in checkpoint['meta']:
#             config = mmengine.Config.fromstring(
#                 checkpoint['meta']['config'], file_format='.py')
#             previous_gpu_ids = config.get('gpu_ids', None)
#             if (previous_gpu_ids is not None and len(previous_gpu_ids) > 0
#                     and len(previous_gpu_ids) != self._world_size):
#                 # TODO, should we modify the iteration?
#                 if (self.auto_scale_lr is None
#                         or not self.auto_scale_lr.get('enable', False)):
#                     raise RuntimeError(
#                         'Number of GPUs used for current experiment is not '
#                         'consistent with the checkpoint being resumed from. '
#                         'This will result in poor performance due to the '
#                         'learning rate. You must set the '
#                         '`auto_scale_lr` parameter for Runner and make '
#                         '`auto_scale_lr["enable"]=True`.')
#                 else:
#                     self.logger.info(
#                         'Number of GPU used for current experiment is not '
#                         'consistent with resuming from checkpoint but the '
#                         'leaning rate will be adjusted according to the '
#                         f'setting in auto_scale_lr={self.auto_scale_lr}')

#         # resume random seed
#         resumed_seed = checkpoint['meta'].get('seed', None)
#         current_seed = self._randomness_cfg.get('seed')
#         if resumed_seed is not None and resumed_seed != current_seed:
#             if current_seed is not None:
#                 self.logger.warning(f'The value of random seed in the '
#                                     f'checkpoint "{resumed_seed}" is '
#                                     f'different from the value in '
#                                     f'`randomness` config "{current_seed}"')
#             self._randomness_cfg.update(seed=resumed_seed)
#             self.set_randomness(**self._randomness_cfg)

#         resumed_dataset_meta = checkpoint['meta'].get('dataset_meta', None)
#         dataset_meta = getattr(self.train_dataloader.dataset, 'metainfo', None)

#         # `resumed_dataset_meta` and `dataset_meta` could be object like
#         # np.ndarray, which cannot be directly judged as equal or not,
#         # therefore we just compared their dumped results.
#         if pickle.dumps(resumed_dataset_meta) != pickle.dumps(dataset_meta):
#             self.logger.warning(
#                 'The dataset metainfo from the resumed checkpoint is '
#                 'different from the current training dataset, please '
#                 'check the correctness of the checkpoint or the training '
#                 'dataset.')

#         self.message_hub.load_state_dict(checkpoint['message_hub'])

#         # resume optimizer
#         if 'optimizer' in checkpoint and resume_optimizer:
#             self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
#             self.optim_wrapper.load_state_dict(  # type: ignore
#                 checkpoint['optimizer'])

#         # resume param scheduler
#         if resume_param_scheduler and self.param_schedulers is None:
#             self.logger.warning(
#                 '`resume_param_scheduler` is True but `self.param_schedulers` '
#                 'is None, so skip resuming parameter schedulers')
#             resume_param_scheduler = False
#         if 'param_schedulers' in checkpoint and resume_param_scheduler:
#             self.param_schedulers = self.build_param_scheduler(  # type: ignore
#                 self.param_schedulers)  # type: ignore
#             if isinstance(self.param_schedulers, dict):
#                 for name, schedulers in self.param_schedulers.items():
#                     for scheduler, ckpt_scheduler in zip(
#                             schedulers, checkpoint['param_schedulers'][name]):
#                         scheduler.load_state_dict(ckpt_scheduler)
#             else:
#                 for scheduler, ckpt_scheduler in zip(
#                         self.param_schedulers,  # type: ignore
#                         checkpoint['param_schedulers']):
#                     scheduler.load_state_dict(ckpt_scheduler)

#         self._has_loaded = True

#         self.logger.info(f'resumed epoch: {self.epoch}, iter: {self.iter}')

#     def load_checkpoint(self,
#                         filename: str,
#                         map_location: Union[str, Callable] = 'cpu',
#                         strict: bool = False,
#                         revise_keys: list = [(r'^module.', '')]):
#         """Load checkpoint from given ``filename``.

#         Args:
#             filename (str): Accept local filepath, URL, ``torchvision://xxx``,
#                 ``open-mmlab://xxx``.
#             map_location (str or callable): A string or a callable function to
#                 specifying how to remap storage locations.
#                 Defaults to 'cpu'.
#             strict (bool): strict (bool): Whether to allow different params for
#                 the model and checkpoint.
#             revise_keys (list): A list of customized keywords to modify the
#                 state_dict in checkpoint. Each item is a (pattern, replacement)
#                 pair of the regular expression operations. Defaults to strip
#                 the prefix 'module.' by [(r'^module\\.', '')].
#         """
#         checkpoint = _load_checkpoint(filename, map_location=map_location)

#         # Add comments to describe the usage of `after_load_ckpt`
#         self.call_hook('after_load_checkpoint', checkpoint=checkpoint)

#         if is_model_wrapper(self.model):
#             model = self.model.module
#         else:
#             model = self.model

#         checkpoint = _load_checkpoint_to_model(
#             model, checkpoint, strict, revise_keys=revise_keys)

#         self._has_loaded = True

#         self.logger.info(f'Load checkpoint from {filename}')

#         return checkpoint

#     @master_only
#     def save_checkpoint(
#         self,
#         out_dir: str,
#         filename: str,
#         file_client_args: Optional[dict] = None,
#         save_optimizer: bool = True,
#         save_param_scheduler: bool = True,
#         meta: Optional[dict] = None,
#         by_epoch: bool = True,
#         backend_args: Optional[dict] = None,
#     ):
#         """Save checkpoints.

#         ``CheckpointHook`` invokes this method to save checkpoints
#         periodically.

#         Args:
#             out_dir (str): The directory that checkpoints are saved.
#             filename (str): The checkpoint filename.
#             file_client_args (dict, optional): Arguments to instantiate a
#                 FileClient. See :class:`mmengine.fileio.FileClient` for
#                 details. Defaults to None. It will be deprecated in future.
#                 Please use `backend_args` instead.
#             save_optimizer (bool): Whether to save the optimizer to
#                 the checkpoint. Defaults to True.
#             save_param_scheduler (bool): Whether to save the param_scheduler
#                 to the checkpoint. Defaults to True.
#             meta (dict, optional): The meta information to be saved in the
#                 checkpoint. Defaults to None.
#             by_epoch (bool): Decide the number of epoch or iteration saved in
#                 checkpoint. Defaults to True.
#             backend_args (dict, optional): Arguments to instantiate the
#                 prefix of uri corresponding backend. Defaults to None.
#                 New in v0.2.0.
#         """
#         if meta is None:
#             meta = {}
#         elif not isinstance(meta, dict):
#             raise TypeError(
#                 f'meta should be a dict or None, but got {type(meta)}')

#         if by_epoch:
#             # self.epoch increments 1 after
#             # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
#             # called by `after_train_epoch`` method of `CheckpointHook` so
#             # `epoch` should be `self.epoch + 1`
#             meta.setdefault('epoch', self.epoch + 1)
#             meta.setdefault('iter', self.iter)
#         else:
#             meta.setdefault('epoch', self.epoch)
#             meta.setdefault('iter', self.iter + 1)

#         if file_client_args is not None:
#             warnings.warn(
#                 '"file_client_args" will be deprecated in future. '
#                 'Please use "backend_args" instead', DeprecationWarning)
#             if backend_args is not None:
#                 raise ValueError(
#                     '"file_client_args" and "backend_args" cannot be set at '
#                     'the same time.')

#             file_client = FileClient.infer_client(file_client_args, out_dir)
#             filepath = file_client.join_path(out_dir, filename)
#         else:
#             filepath = join_path(  # type: ignore
#                 out_dir, filename, backend_args=backend_args)

#         meta.update(
#             cfg=self.cfg.pretty_text,
#             seed=self.seed,
#             experiment_name=self.experiment_name,
#             time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
#             mmengine_version=mmengine.__version__ + get_git_hash())

#         if hasattr(self.train_dataloader.dataset, 'metainfo'):
#             meta.update(dataset_meta=self.train_dataloader.dataset.metainfo)

#         if is_model_wrapper(self.model):
#             model = self.model.module
#         else:
#             model = self.model

#         checkpoint = {
#             'meta':
#             meta,
#             'state_dict':
#             weights_to_cpu(model.state_dict()),
#             'message_hub':
#             apply_to(self.message_hub.state_dict(),
#                      lambda x: hasattr(x, 'cpu'), lambda x: x.cpu()),
#         }
#         # save optimizer state dict to checkpoint
#         if save_optimizer:
#             if isinstance(self.optim_wrapper, OptimWrapper):
#                 checkpoint['optimizer'] = apply_to(
#                     self.optim_wrapper.state_dict(),
#                     lambda x: hasattr(x, 'cpu'), lambda x: x.cpu())
#             else:
#                 raise TypeError(
#                     'self.optim_wrapper should be an `OptimWrapper` '
#                     'or `OptimWrapperDict` instance, but got '
#                     f'{self.optim_wrapper}')

#         # save param scheduler state dict
#         if save_param_scheduler and self.param_schedulers is None:
#             self.logger.warning(
#                 '`save_param_scheduler` is True but `self.param_schedulers` '
#                 'is None, so skip saving parameter schedulers')
#             save_param_scheduler = False
#         if save_param_scheduler:
#             if isinstance(self.param_schedulers, dict):
#                 checkpoint['param_schedulers'] = dict()
#                 for name, schedulers in self.param_schedulers.items():
#                     checkpoint['param_schedulers'][name] = []
#                     for scheduler in schedulers:
#                         state_dict = scheduler.state_dict()
#                         checkpoint['param_schedulers'][name].append(state_dict)
#             else:
#                 checkpoint['param_schedulers'] = []
#                 for scheduler in self.param_schedulers:  # type: ignore
#                     state_dict = scheduler.state_dict()  # type: ignore
#                     checkpoint['param_schedulers'].append(state_dict)

#         self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
#         save_checkpoint(
#             checkpoint,
#             filepath,
#             file_client_args=file_client_args,
#             backend_args=backend_args)

#     @master_only
#     def dump_config(self) -> None:
#         """Dump config to `work_dir`."""
#         if self.cfg.filename is not None:
#             filename = osp.basename(self.cfg.filename)
#         else:
#             filename = f'{self.timestamp}.py'
#         self.cfg.dump(osp.join(self.work_dir, filename))

#     def _check_scheduler_cfg(
#             self, param_scheduler: Optional[Union[dict, list,
#                                                   _ParamScheduler]]) -> None:
#         """Parse `param_scheduler` to a list of parameter schedulers, or a
#         `dict` of which each value is a list of parameter schedulers.

#         If only one optimizer is used, the parsed config should be a
#         list of parameter scheduler configs or instances. If multiple
#         optimizers are used, the parsed config should be `dict`.
#         Its key should be consistent with the optimizer `dict` and its value
#         should be a list of parameter scheduler configs or instances. See
#         :meth:`build_param_scheduler` for more details.

#         Examples:
#             >>> # valid scheduler:
#             >>> # empty scheduler
#             >>> scheduler = None
#             >>> # Single scheduler
#             >>> scheduler = dict(type='MultiStepLR', milestones=[1, 2])
#             >>> # Single list schedulers
#             >>> scheduler = [dict(type='MultiStepLR', milestones=[1, 2]),
#             >>>              dict(type='MultiStepLR', milestones=[2, 3])]
#             >>> # `dict` of schedulers
#             >>> scheduler = dict(linear1=dict(type='MultiStepLR', milestones=[1, 2]),
#             >>>                  linear2=dict(type='MultiStepLR', milestones=[1, 2]))
#             >>> # `dict` of `list` of schedulers
#             >>> scheduler = dict(linear1=[dict(type='MultiStepLR', milestones=[1, 2])],
#             >>>                  linear2=[dict(type='MultiStepLR', milestones=[1, 2])])
#             >>> # Single built scheduler
#             >>> from mmengine.optim import MultiStepLR
#             >>> scheduler = MultiStepLR(milestones=[1, 2], optimizer=optimizer)
#             >>> # Single built list schedulers
#             >>> scheduler = [MultiStepLR(milestones=[1, 2], optimizer=optimizer)]
#             >>> # dict of built scheduler
#             >>> scheduler = dict(linear1=MultiStepLR(milestones=[1, 2], optimizer=optimizer),
#             >>>                  linear2=MultiStepLR(milestones=[1, 2], optimizer=optimizer))
#             >>> # dict of built list schedulers
#             >>> scheduler = dict(linear1=[MultiStepLR(milestones=[1, 2], optimizer=optimizer)],
#             >>>                  linear2=[MultiStepLR(milestones=[1, 2], optimizer=optimizer)])

#         Args:
#             param_scheduler (dict or list): The original parameter scheduler.
#         """  # noqa: E501
#         if param_scheduler is None:
#             return
#         if isinstance(param_scheduler, _ParamScheduler):
#             return
#         if is_seq_of(param_scheduler, _ParamScheduler):
#             return

#         if is_seq_of(param_scheduler, dict):
#             for _param_scheduler in param_scheduler:
#                 assert 'type' in _param_scheduler, (
#                     'Each parameter scheduler should contain the key type, '
#                     f'but got {_param_scheduler}')
#         elif isinstance(param_scheduler, dict):
#             if 'type' not in param_scheduler:
#                 for key, _param_scheduler in param_scheduler.items():
#                     assert isinstance(
#                         _param_scheduler,
#                         (dict, tuple, list, _ParamScheduler)), (
#                             'Each value of `param_scheduler` should be a '
#                             f'dict or a list, but got {_param_scheduler} with '
#                             f'type {type(_ParamScheduler)}')

#         else:
#             raise TypeError(
#                 '`param_scheduler` should be a `_ParamScheduler`, `dict`, '
#                 f'list or a tuple, but got {type(param_scheduler)}. If '
#                 '`param_scheduler` is a list of dict, it means a list of '
#                 'scheduler configs for single optimizer. If it is a dict and '
#                 'contains key `type`, it means a scheduler config for a '
#                 'single optimizer. If it does not contain key `type`, it '
#                 'means multiple lists of schedulers for multiple optimizers.')

#     def _log_env(self, env_cfg: dict) -> None:
#         """Logging environment information of the current task.

#         Args:
#             env_cfg (dict): The environment config of the runner.
#         """
#         # Collect and log environment information.
#         env = collect_env()
#         runtime_env = OrderedDict()
#         runtime_env.update(env_cfg)
#         runtime_env.update(self._randomness_cfg)
#         runtime_env['seed'] = self._seed
#         runtime_env['Distributed launcher'] = self._launcher
#         runtime_env['Distributed training'] = self._distributed
#         runtime_env['GPU number'] = self._world_size

#         env_info = '\n    ' + '\n    '.join(f'{k}: {v}'
#                                             for k, v in env.items())
#         runtime_env_info = '\n    ' + '\n    '.join(
#             f'{k}: {v}' for k, v in runtime_env.items())
#         dash_line = '-' * 60
#         self.logger.info('\n' + dash_line + '\nSystem environment:' +
#                          env_info + '\n'
#                          '\nRuntime environment:' + runtime_env_info + '\n' +
#                          dash_line + '\n')

#         if self.cfg._cfg_dict:
#             self.logger.info(f'Config:\n{self.cfg.pretty_text}')

#     def _maybe_compile(self, target: str) -> None:
#         """Use `torch.compile` to optimize model/wrapped_model."""
#         compile_cfg = self.cfg.get('compile', None)
#         if compile_cfg is None:
#             # no compile options given, won't compile
#             return

#         if isinstance(compile_cfg, bool):
#             if not compile_cfg:
#                 # compile=False, compilation is disabled
#                 return
#             # compile=True, use default configurations
#             compile_cfg = dict()

#         assert digit_version(TORCH_VERSION) >= digit_version('2.0.0'), (
#             'PyTorch >= 2.0.0 is required to enable torch.compile')
#         assert isinstance(compile_cfg, dict), (
#             f'`compile` should be a dict or bool, got {type(compile_cfg)}')

#         func = getattr(self.model, target)
#         compiled_func = torch.compile(func, **compile_cfg)
#         setattr(self.model, target, compiled_func)
#         self.logger.info('Model has been "compiled". The first few iterations'
#                          ' will be slow, please be patient.')

# # Copyright (c) OpenMMLab. All rights reserved.
# import copy
# import datetime
# import re
# from collections import OrderedDict
# from itertools import chain
# from typing import List, Optional, Tuple

# import numpy as np
# import torch

# from mmengine.device import get_max_cuda_memory, is_cuda_available


# @LOG_PROCESSORS.register_module()
# class LogProcessor:
#     """A log processor used to format log information collected from
#     ``runner.message_hub.log_scalars``.

#     ``LogProcessor`` instance is built by runner and will format
#     ``runner.message_hub.log_scalars`` to ``tag`` and ``log_str``, which can
#     directly used by ``LoggerHook`` and ``MMLogger``. Besides, the argument
#     ``custom_cfg`` of constructor can control the statistics method of logs.

#     Args:
#         window_size (int): default smooth interval. Defaults to 10.
#         by_epoch (bool): Whether to format logs with epoch stype. Defaults to
#             True.
#         custom_cfg (list[dict], optional): Contains multiple log config dict,
#             in which key means the data source name of log and value means the
#             statistic method and corresponding arguments used to count the
#             data source. Defaults to None.

#             - If custom_cfg is None, all logs will be formatted via default
#               methods, such as smoothing loss by default window_size. If
#               custom_cfg is defined as a list of config dict, for example:
#               [dict(data_src='loss', method='mean', log_name='global_loss',
#               window_size='global')]. It means the log item ``loss`` will be
#               counted as global mean and additionally logged as ``global_loss``
#               (defined by ``log_name``). If ``log_name`` is not defined in
#               config dict, the original logged key will be overwritten.

#             - The original log item cannot be overwritten twice. Here is
#               an error example:
#               [dict(data_src='loss', method='mean', window_size='global'),
#               dict(data_src='loss', method='mean', window_size='epoch')].
#               Both log config dict in custom_cfg do not have ``log_name`` key,
#               which means the loss item will be overwritten twice.

#             - For those statistic methods with the ``window_size`` argument,
#               if ``by_epoch`` is set to False, ``windows_size`` should not be
#               `epoch` to statistics log value by epoch.
#         num_digits (int): The number of significant digit shown in the
#             logging message. Defaults to 4.
#         log_with_hierarchy (bool): Whether to log with hierarchy. If it is
#             True, the information is written to visualizer backend such as
#             :obj:`LocalVisBackend` and :obj:`TensorboardBackend`
#             with hierarchy. For example, ``loss`` will be saved as
#             ``train/loss``, and accuracy will be saved as ``val/accuracy``.
#             Defaults to False.
#             `New in version 0.7.0.`
#         mean_pattern (str): This is a regular expression used to match the log
#             that need to be included in the smoothing statistics.
#             `New in version 0.7.3.`

#     Examples:
#         >>> # `log_name` is defined, `loss_large_window` will be an additional
#         >>> # record.
#         >>> log_processor = dict(
#         >>>     window_size=10,
#         >>>     by_epoch=True,
#         >>>     custom_cfg=[dict(data_src='loss',
#         >>>                       log_name='loss_large_window',
#         >>>                       method_name='mean',
#         >>>                       window_size=100)])
#         >>> # `log_name` is not defined. `loss` will be overwritten.
#         >>> log_processor = dict(
#         >>>     window_size=10,
#         >>>     by_epoch=True,
#         >>>     custom_cfg=[dict(data_src='loss',
#         >>>                       method_name='mean',
#         >>>                       window_size=100)])
#         >>> # Record loss with different statistics methods.
#         >>> log_processor = dict(
#         >>>     window_size=10,
#         >>>     by_epoch=True,
#         >>>     custom_cfg=[dict(data_src='loss',
#         >>>                       log_name='loss_large_window',
#         >>>                       method_name='mean',
#         >>>                       window_size=100),
#         >>>                  dict(data_src='loss',
#         >>>                       method_name='mean',
#         >>>                       window_size=100)])
#         >>> # Overwrite loss item twice will raise an error.
#         >>> log_processor = dict(
#         >>>     window_size=10,
#         >>>     by_epoch=True,
#         >>>     custom_cfg=[dict(data_src='loss',
#         >>>                       method_name='mean',
#         >>>                       window_size=100),
#         >>>                  dict(data_src='loss',
#         >>>                       method_name='max',
#         >>>                       window_size=100)])
#         AssertionError
#     """

#     def __init__(self,
#                  window_size=10,
#                  by_epoch=True,
#                  custom_cfg: Optional[List[dict]] = None,
#                  num_digits: int = 4,
#                  log_with_hierarchy: bool = False,
#                  mean_pattern=r'.*(loss|time|data_time|grad_norm).*'):
#         self.window_size = window_size
#         self.by_epoch = by_epoch
#         self.custom_cfg = custom_cfg if custom_cfg else []
#         self.num_digits = num_digits
#         self.log_with_hierarchy = log_with_hierarchy
#         self.mean_pattern = re.compile(mean_pattern)
#         self._check_custom_cfg()

#     def get_log_after_iter(self, runner, batch_idx: int,
#                            mode: str) -> Tuple[dict, str]:
#         """Format log string after training, validation or testing iteration.

#         Args:
#             runner (Runner): The runner of training phase.
#             batch_idx (int): The index of the current batch in the current
#                 loop.
#             mode (str): Current mode of runner, train, test or val.

#         Return:
#             Tuple[dict, str]: Formatted log dict/string which will be
#             recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
#         """
#         assert mode in ['train', 'test', 'val']
#         # Overwrite ``window_size`` defined in ``custom_cfg`` to int value.
#         parsed_cfg = self._parse_windows_size(runner, batch_idx,
#                                               self.custom_cfg)
#         # log_tag is used to write log information to terminal
#         log_tag = self._collect_scalars(parsed_cfg, runner, mode)

#         # If `self.log_with_hierarchy` is False, the tag is the same as
#         # log_tag. Otherwise, each key in tag starts with prefix `train`,
#         # `test` or `val`
#         if not self.log_with_hierarchy:
#             tag = copy.deepcopy(log_tag)
#         else:
#             tag = self._collect_scalars(parsed_cfg, runner, mode, True)

#         # Record learning rate.
#         lr_str_list = []
#         for key, value in tag.items():
#             if key.endswith('lr'):
#                 key = self._remove_prefix(key, f'{mode}/')
#                 log_tag.pop(key)
#                 lr_str_list.append(f'{key}: '
#                                    f'{value:.{self.num_digits}e}')
#         lr_str = ' '.join(lr_str_list)
#         # Format log header.
#         # by_epoch == True
#         #   train/val: Epoch [5][5/10]  ...
#         #   test: Epoch [5/10]
#         # by_epoch == False
#         #  train: Epoch [5/10000] ... (divided by `max_iter`)
#         #  val/test: Epoch [5/2000] ... (divided by length of dataloader)
#         if self.by_epoch:
#             # Align the iteration log:
#             # Epoch(train)  [  9][010/270]
#             # ...                 ||| |||
#             # Epoch(train)  [ 10][100/270]
#             dataloader_len = self._get_dataloader_size(runner, mode)
#             cur_iter = self._get_iter(runner, batch_idx)
#             cur_iter_str = str(cur_iter).rjust(len(str(dataloader_len)))
#             if mode in ['train', 'val']:
#                 cur_epoch = self._get_epoch(runner, mode)
#                 if not (isinstance(runner._train_loop, dict)
#                         or runner._train_loop is None):
#                     # Right Align the epoch log:
#                     # Epoch(train)   [9][100/270]
#                     # ...             ||
#                     # Epoch(train) [100][100/270]
#                     max_epochs = runner.max_epochs
#                     # 3 means the three characters: "[", "]", and " " occupied
#                     # in " [{max_epochs}]"
#                     cur_epoch_str = f'[{cur_epoch}]'.rjust(
#                         len(str(max_epochs)) + 3, ' ')
#                 else:
#                     cur_epoch_str = f'[{cur_epoch}]'
#                 tag['epoch'] = cur_epoch
#                 log_str = (f'Epoch({mode}){cur_epoch_str}'
#                            f'[{cur_iter_str}/{dataloader_len}]  ')
#             else:
#                 log_str = (f'Epoch({mode}) '
#                            f'[{cur_iter_str}/{dataloader_len}]  ')
#         else:
#             if mode == 'train':
#                 cur_iter = self._get_iter(runner, batch_idx)
#                 cur_iter_str = str(cur_iter).rjust(len(str(runner.max_iters)))
#                 log_str = (f'Iter({mode}) '
#                            f'[{cur_iter_str}/{runner.max_iters}]  ')
#             else:
#                 dataloader_len = self._get_dataloader_size(runner, mode)
#                 cur_iter_str = str(batch_idx + 1).rjust(
#                     len(str(dataloader_len)))
#                 log_str = (f'Iter({mode}) [{cur_iter_str}/{dataloader_len}]  ')
        
#         # Add global iter.
#         if isinstance(runner._train_loop, dict) or runner._train_loop is None:
#             tag['iter'] = 0
#         else:
#             tag['iter'] = runner.iter + 1
#         # Concatenate lr, momentum string with log header.
#         log_str += f'{lr_str}  '
#         # If IterTimerHook used in runner, eta, time, and data_time should be
#         # recorded.
#         if (all(item in log_tag for item in ['time', 'data_time'])
#                 and 'eta' in runner.message_hub.runtime_info):
#             eta = runner.message_hub.get_info('eta')
#             eta_str = str(datetime.timedelta(seconds=int(eta)))
#             log_str += f'eta: {eta_str}  '
#             log_str += (f'time: {log_tag["time"]:.{self.num_digits}f}  '
#                         f'data_time: '
#                         f'{log_tag["data_time"]:.{self.num_digits}f}  ')
#             # Pop recorded keys
#             log_tag.pop('time')
#             log_tag.pop('data_time')

#         # If cuda is available, the max memory occupied should be calculated.
#         if is_cuda_available():
#             max_memory = self._get_max_memory(runner)
#             log_str += f'memory: {max_memory}  '
#             tag['memory'] = max_memory
#         # Loop left keys to fill `log_str`.
#         if mode in ('train', 'val'):
#             log_items = []
#             for name, val in log_tag.items():
#                 # if mode == 'val' and not name.startswith('val/loss'):
#                 if mode == 'val':
#                     val_loss = runner.message_hub.log_scalars.get('val/loss', None)
#                     # print("val_loss",val_loss)
#                     if val_loss is not None:
#                         averaged_val_loss = val_loss.mean(batch_idx + 1)
#                         log_str += f'val_loss: {averaged_val_loss:.{self.num_digits}f}  '
#                         tag['val_loss'] = averaged_val_loss
#                 if isinstance(val, float):
#                     val = f'{val:.{self.num_digits}f}'
#                 log_items.append(f'{name}: {val}')
#             log_str += '  '.join(log_items)
#         return tag, log_str

#     def get_log_after_epoch(self,
#                             runner,
#                             batch_idx: int,
#                             mode: str,
#                             with_non_scalar: bool = False) -> Tuple[dict, str]:
#         """Format log string after validation or testing epoch.

#         Args:
#             runner (Runner): The runner of validation/testing phase.
#             batch_idx (int): The index of the current batch in the current
#                 loop.
#             mode (str): Current mode of runner.
#             with_non_scalar (bool): Whether to include non-scalar infos in the
#                 returned tag. Defaults to False.

#         Return:
#             Tuple[dict, str]: Formatted log dict/string which will be
#             recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
#         """
#         assert mode in [
#             'test', 'val'
#         ], ('`_get_metric_log_str` only accept val or test mode, but got '
#             f'{mode}')
#         dataloader_len = self._get_dataloader_size(runner, mode)
        
#         # By epoch:
#         #     Epoch(val) [10][1000/1000]  ...
#         #     Epoch(test) [1000/1000] ...
#         # By iteration:
#         #     Iteration(val) [1000/1000]  ...
#         #     Iteration(test) [1000/1000]  ...
#         if self.by_epoch:
#             if mode == 'val':
#                 cur_epoch = self._get_epoch(runner, mode)
#                 val_loss = runner.message_hub.log_scalars.get('val/loss', None)
#                 # print("val_loss",val_loss)
#                 if val_loss is not None:
#                     averaged_val_loss = val_loss.mean(batch_idx + 1)
#                     # print("averaged_val_loss",averaged_val_loss)
#                     log_str = (f'Epoch({mode}) [{cur_epoch}][{dataloader_len}/'
#                                f'{dataloader_len}]  '
#                                f'val_loss: {averaged_val_loss:.{self.num_digits}f}  ')
#                     tag['val_loss'] = averaged_val_loss
#                 else:
#                     log_str = (f'Epoch({mode}) [{cur_epoch}][{dataloader_len}/'
#                            f'{dataloader_len}]  ')
#             else:
#                 log_str = (
#                     f'Epoch({mode}) [{dataloader_len}/{dataloader_len}]  ')

#         else:
#             # log_str = (f'Iter({mode}) [{dataloader_len}/{dataloader_len}]  ')
#             # by_epoch가 False인 경우의 log_str 초기화
#             log_str = (f'Iter({mode}) [{dataloader_len}/{dataloader_len}]  ')        
        
        
#         custom_cfg_copy = copy.deepcopy(self.custom_cfg)
#         # remove prefix
#         custom_keys = [
#             self._remove_prefix(cfg['data_src'], f'{mode}/')
#             for cfg in custom_cfg_copy
#         ]
#         # Count the averaged time and data_time by epoch
#         if 'time' not in custom_keys:
#             custom_cfg_copy.append(
#                 dict(data_src='time', window_size='epoch', method_name='mean'))
#         if 'data_time' not in custom_keys:
#             custom_cfg_copy.append(
#                 dict(
#                     data_src='data_time',
#                     window_size='epoch',
#                     method_name='mean'))
#         parsed_cfg = self._parse_windows_size(runner, batch_idx,
#                                               custom_cfg_copy)
#         # tag is used to write log information to different backends.
#         ori_tag = self._collect_scalars(parsed_cfg, runner, mode,
#                                         self.log_with_hierarchy)
#         non_scalar_tag = self._collect_non_scalars(runner, mode)
#         # move `time` or `data_time` to the end of the log
#         tag = OrderedDict()
#         time_tag = OrderedDict()
#         for key, value in ori_tag.items():
#             if key in (f'{mode}/time', f'{mode}/data_time', 'time',
#                        'data_time'):
#                 time_tag[key] = value
#             else:
#                 tag[key] = value
#         # Log other messages.
#         log_items = []
#         log_str += '  '
#         for name, val in chain(tag.items(), non_scalar_tag.items(),
#                                time_tag.items()):
#             if isinstance(val, float):
#                 val = f'{val:.{self.num_digits}f}'
#             if isinstance(val, (torch.Tensor, np.ndarray)):
#                 # newline to display tensor and array.
#                 val = f'\n{val}\n'
#             log_items.append(f'{name}: {val}')
#         log_str += '  '.join(log_items)

#         if with_non_scalar:
#             tag.update(non_scalar_tag)
#         tag.update(time_tag)
#         return tag, log_str

#     def _collect_scalars(self,
#                          custom_cfg: List[dict],
#                          runner,
#                          mode: str,
#                          reserve_prefix: bool = False) -> dict:
#         """Collect log information to compose a dict according to mode.

#         Args:
#             custom_cfg (List[dict]): A copy of ``self.custom_cfg`` with int
#                 ``window_size``.
#             runner (Runner): The runner of the training/testing/validation
#                 process.
#             mode (str): Current mode of runner.
#             reserve_prefix (bool): Whether to reserve the prefix of the key.

#         Returns:
#             dict: Statistical values of logs.
#         """
#         custom_cfg = copy.deepcopy(custom_cfg)
#         tag = OrderedDict()
#         # history_scalars of train/val/test phase.
#         history_scalars = runner.message_hub.log_scalars
#         # corresponding mode history_scalars
#         mode_history_scalars = OrderedDict()
#         # extract log scalars and remove prefix to `mode_history_scalars`
#         # according to mode.
#         for prefix_key, log_buffer in history_scalars.items():
#             if prefix_key.startswith(mode):
#                 if not reserve_prefix:
#                     key = self._remove_prefix(prefix_key, f'{mode}/')
#                 else:
#                     key = prefix_key
#                 mode_history_scalars[key] = log_buffer
#         for key in mode_history_scalars:
#             # Update the latest learning rate and smoothed time logs.
#             if re.search(self.mean_pattern, key) is not None:
#                 tag[key] = mode_history_scalars[key].mean(self.window_size)
#             else:
#                 # Default statistic method is current.
#                 tag[key] = mode_history_scalars[key].current()
#         # Update custom keys.
#         for log_cfg in custom_cfg:
#             data_src = log_cfg.pop('data_src')
#             log_name = log_cfg.pop('log_name', data_src)
#             if reserve_prefix:
#                 data_src = f'{mode}/{data_src}'
#                 log_name = f'{mode}/{log_name}'
#             # log item in custom_cfg could only exist in train or val
#             # mode.
#             if data_src in mode_history_scalars:
#                 tag[log_name] = mode_history_scalars[data_src].statistics(
#                     **log_cfg)
#         return tag

#     def _collect_non_scalars(self, runner, mode: str) -> dict:
#         """Collect log information to compose a dict according to mode.

#         Args:
#             runner (Runner): The runner of the training/testing/validation
#                 process.
#             mode (str): Current mode of runner.

#         Returns:
#             dict: non-scalar infos of the specified mode.
#         """
#         # infos of train/val/test phase.
#         infos = runner.message_hub.runtime_info
#         # corresponding mode infos
#         mode_infos = OrderedDict()
#         # extract log info and remove prefix to `mode_infos` according to mode.
#         for prefix_key, value in infos.items():
#             if prefix_key.startswith(mode):
#                 if self.log_with_hierarchy:
#                     key = prefix_key
#                 else:
#                     key = self._remove_prefix(prefix_key, f'{mode}/')
#                 mode_infos[key] = value
#         return mode_infos

#     def _remove_prefix(self, string: str, prefix: str):
#         """Remove the prefix ``train``, ``val`` and ``test`` of the key."""
#         if string.startswith(prefix):
#             return string[len(prefix):]
#         else:
#             return string

#     def _check_custom_cfg(self) -> None:
#         """Check the legality of ``self.custom_cfg``."""

#         def _check_window_size():
#             for log_cfg in self.custom_cfg:
#                 if not self.by_epoch:
#                     assert log_cfg['window_size'] != 'epoch', \
#                         'window_size cannot be epoch if LoggerHook.by_epoch' \
#                         ' is False.'

#         def _check_repeated_log_name():
#             # The `log_name` of the same data_src should not be repeated.
#             # If `log_name` is not specified, `data_src` will be overwritten.
#             # But only allowed to be overwritten once.
#             check_set = set()
#             for log_cfg in self.custom_cfg:
#                 assert 'data_src' in log_cfg
#                 data_src = log_cfg['data_src']
#                 log_name = log_cfg.get('log_name', data_src)
#                 assert log_name not in check_set, (
#                     f'Found duplicate {log_name} for {data_src}. Please check'
#                     'your `custom_cfg` for `log_processor`. You should '
#                     f'neither define duplicate `{log_name}` for {data_src} '
#                     f'nor do not define any {log_name} for multiple '
#                     f'{data_src}, See more information in the docstring of '
#                     'LogProcessor')

#                 check_set.add(log_name)

#         _check_repeated_log_name()
#         _check_window_size()

#     def _parse_windows_size(self,
#                             runner,
#                             batch_idx: int,
#                             custom_cfg: Optional[list] = None) -> list:
#         """Parse window_size defined in custom_cfg to int value.

#         Args:
#             runner (Runner): The runner of the training/testing/validation
#                 process.
#             batch_idx (int): The iteration index of current dataloader.
#             custom_cfg (list): A copy of ``self.custom_cfg``. Defaults to None
#                 to keep backward compatibility.
#         """
#         if custom_cfg is None:
#             custom_cfg = copy.deepcopy(self.custom_cfg)
#         else:
#             custom_cfg = copy.deepcopy(custom_cfg)
#         for log_cfg in custom_cfg:
#             window_size = log_cfg.get('window_size', None)
#             if window_size is None or isinstance(window_size, int):
#                 continue
#             elif window_size == 'epoch':
#                 log_cfg['window_size'] = batch_idx + 1
#             elif window_size == 'global':
#                 log_cfg['window_size'] = runner.iter + 1
#             else:
#                 raise TypeError(
#                     'window_size should be int, epoch or global, but got '
#                     f'invalid {window_size}')
#         return custom_cfg

#     def _get_max_memory(self, runner) -> int:
#         """Returns the maximum GPU memory occupied by tensors in megabytes (MB)
#         for a given device.

#         Args:
#             runner (Runner): The runner of the training/testing/validation
#                 process.

#         Returns:
#             The maximum GPU memory occupied by tensors in megabytes for a given
#             device.
#         """

#         device = getattr(runner.model, 'output_device', None)
#         return get_max_cuda_memory(device)

#     def _get_iter(self, runner, batch_idx: int) -> int:
#         """Get current iteration index.

#         Args:
#             runner (Runner): The runner of the training/testing/validation
#                 process.
#             batch_idx (int): The iteration index of current
#                 dataloader. Defaults to None.

#         Returns:
#             int: The current global iter or inner iter.
#         """
#         if self.by_epoch:
#             current_iter = batch_idx + 1
#         else:
#             current_iter = runner.iter + 1
#         return current_iter

#     def _get_epoch(self, runner, mode: str) -> int:
#         """Get current epoch according to mode.

#         Args:
#             runner (Runner): The runner of the training/testing/validation
#                 process.
#             mode (str): Current mode of runner.

#         Returns:
#             int: The current epoch.
#         """
#         if mode == 'train':
#             epoch = runner.epoch + 1
#         elif mode == 'val':
#             if (isinstance(runner._train_loop, dict)
#                     or runner._train_loop is None):
#                 epoch = 0
#             else:
#                 # normal val mode
#                 # runner.epoch += 1 has been done before validation
#                 epoch = runner.epoch
#         else:
#             raise ValueError(
#                 f"runner mode should be 'train' or 'val', but got {mode}")
#         return epoch

#     def _get_cur_loop(self, runner, mode: str):
#         """Get current loop according to mode.

#         Args:
#             runner (Runner): The runner of the training/validation/testing
#                 process.
#             mode (str): Current mode of runner.

#         Returns:
#             BaseLoop: Current loop of runner.
#         """
#         # returns type hint will occur circular import
#         if mode == 'train':
#             return runner.train_loop
#         elif mode == 'val':
#             return runner.val_loop
#         else:
#             return runner.test_loop

#     def _get_dataloader_size(self, runner, mode) -> int:
#         """Get dataloader size of current loop.

#         Args:
#             runner (Runner): The runner of the training/validation/testing
#             mode (str): Current mode of runner.

#         Returns:
#             int: The dataloader size of current loop.
#         """
#         return len(self._get_cur_loop(runner=runner, mode=mode).dataloader)
# @LOOPS.register_module()
# class ValLoop(BaseLoop):
#     """Loop for validation.

#     Args:
#         runner (Runner): A reference of runner.
#         dataloader (Dataloader or dict): A dataloader object or a dict to
#             build a dataloader.
#         evaluator (Evaluator or dict or list): Used for computing metrics.
#         fp16 (bool): Whether to enable fp16 validation. Defaults to
#             False.
#     """

#     def __init__(self,
#                  runner,
#                  dataloader: Union[DataLoader, Dict],
#                  evaluator: Union[Evaluator, Dict, List],
#                  fp16: bool = False,
#                  max_epochs: int =100) -> None:
#         super().__init__(runner, dataloader)
#         self._max_epochs = int(max_epochs)
#         assert self._max_epochs == max_epochs, \
#             f'`max_epochs` should be a integer number, but get {max_epochs}.'
#         # self._max_epochs = max_epochs
#         self._epoch = 0
#         self._max_iters = self._max_epochs * len(self.dataloader)
#         self._iter = 0  
#         if isinstance(evaluator, (dict, list)):
#             self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
#         else:
#             assert isinstance(evaluator, Evaluator), (
#                 'evaluator must be one of dict, list or Evaluator instance, '
#                 f'but got {type(evaluator)}.')
#             self.evaluator = evaluator  # type: ignore
#         if hasattr(self.dataloader.dataset, 'metainfo'):
#             self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
#             self.runner.visualizer.dataset_meta = \
#                 self.dataloader.dataset.metainfo
#         else:
#             print_log(
#                 f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
#                 'metainfo. ``dataset_meta`` in evaluator, metric and '
#                 'visualizer will be None.',
#                 logger='current',
#                 level=logging.WARNING)
#         self.fp16 = fp16
#     @property
#     def max_epochs(self):
#         """int: Total epochs for validation."""
#         return self._max_epochs

#     @property
#     def epoch(self):
#         """int: Current validation epoch."""
#         return self._epoch
#     # def run(self) -> dict:
#     #     """Launch validation."""
#     #     self.runner.call_hook('before_val')
#     #     self.runner.call_hook('before_val_epoch')
#     #     self.runner.model.eval()
#     #     for idx, data_batch in enumerate(self.dataloader):
#     #         self.run_iter(idx, data_batch)

#     #     # compute metrics
#     #     metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
#     #     self.runner.call_hook('after_val_epoch', metrics=metrics)
#     #     self.runner.call_hook('after_val')
#     #     return metrics
#     def run(self) -> dict:
#         """Launch validation with epoch-based logic."""
#         self.runner.call_hook('before_val')
#         self.runner.call_hook('before_val_epoch')
#         self.runner.model.eval()
        
#         for idx, data_batch in enumerate(self.dataloader):
#             self.run_iter(idx, data_batch)
#         while self._epoch < self._max_epochs:
#             self._epoch += 1
#         metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
#         self.runner.call_hook('after_val_epoch', metrics=metrics)
#         self.runner.call_hook('after_val')
#         return metrics

#     # def run_epoch(self) -> dict:
#     #     """Iterate one epoch for validation."""
#     #     self.runner.call_hook('before_val_epoch')
#     #     self.runner.model.eval()

#     #     for idx, data_batch in enumerate(self.dataloader):
#     #         self.run_iter(idx, data_batch)

#     #     metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
#     #     self.runner.call_hook('after_val_epoch', metrics=metrics)
#     #     return metrics

#     @torch.no_grad()
#     def run_iter(self, idx, data_batch: Sequence[dict]):
#         """Iterate one mini-batch.

#         Args:
#             data_batch (Sequence[dict]): Batch of data
#                 from dataloader.
#         """
#         self.runner.call_hook(
#             'before_val_iter', batch_idx=idx, data_batch=data_batch)
#         # outputs should be sequence of BaseDataElement
#         with autocast(enabled=self.fp16):
#             outputs = self.runner.model.val_step(data_batch)
#         self.evaluator.process(data_samples=outputs, data_batch=data_batch)
#         self.runner.call_hook(
#             'after_val_iter',
#             batch_idx=idx,
#             data_batch=data_batch,
#             outputs=outputs)


@MMENGINE_DATA_SAMPLERS.register_module()
class CustomSampler(Sampler):
    """The default data sampler for both distributed and non-distributed
    environment.

    It has several differences from the PyTorch ``DistributedSampler`` as
    below:

    1. This sampler supports non-distributed environment.

    2. The round up behaviors are a little different.

       - If ``round_up=True``, this sampler will add extra samples to make the
         number of samples is evenly divisible by the world size. And
         this behavior is the same as the ``DistributedSampler`` with
         ``drop_last=False``.
       - If ``round_up=False``, this sampler won't remove or add any samples
         while the ``DistributedSampler`` with ``drop_last=True`` will remove
         tail samples.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True,
                 ann_file: str = '') -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)
        
        df = pd.DataFrame(open(ann_file).readlines(), columns=['path']).reset_index()
        df['path'] = df.loc[:, 'path'].apply(lambda x: osp.split(x.split()[0])[0])
        self.grouped_data = df.groupby('path')['index'].apply(list).to_dict()
        self.num_samples = len(self.grouped_data)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        indices = []
        for idx in self.grouped_data.values():
            indices.append(random.choice(idx))
        if self.shuffle:
            random.shuffle(indices)
        print(indices)
        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples
    
@MMENGINE_DATA_SAMPLERS.register_module()
class LimitedSampleSampler(Sampler):
    """Sampler that only uses a limited number of samples per epoch."""

    def __init__(self, dataset: Sized, num_samples: int = 20, shuffle: bool = True, seed: Optional[int] = None):
        self.dataset = dataset
        self.num_samples = num_samples
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        if self.shuffle:
            random.seed(self.seed)
            indices = random.sample(range(len(self.dataset)), self.num_samples)
        else:
            indices = list(range(self.num_samples))

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples