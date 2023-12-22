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
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.utils import calc_dynamic_intervals
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

@LOOPS.register_module()
class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False,
                 print_loss: bool = True) -> None:
        super().__init__(runner, dataloader)

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16
        # custom
        self.print_loss = print_loss 
        
    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        # custom
        val_loss = 0.0
        
        for idx, data_batch in enumerate(self.dataloader):
                    loss = self.run_iter(idx, data_batch)  # Get the validation loss
                    val_loss += loss  # Accumulate the validation loss
        avg_val_loss = val_loss / len(self.dataloader)  # Calculate average validation loss

        if self.print_loss:
            print(f"Average Validation Loss: {avg_val_loss}")  # Print the validation loss
        
        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs_list = self.runner.model.val_step(data_batch)
            
        val_loss = 0.0
        for outputs in outputs_list:
            loss = outputs.get('loss', 0.0)  # Access 'loss' key from each dictionary
            val_loss += loss  # Accumulate the validation loss
        
        # Assuming self.evaluator.process() expects a list of data samples
        # Convert outputs_list to a list of dictionaries if needed
        processed_samples = [{'loss': output.get('loss', 0.0)} for output in outputs_list]

        self.evaluator.process(data_samples=processed_samples, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs_list)
        
        return val_loss / len(outputs_list) if outputs_list else 0.0


@LOG_PROCESSORS.register_module()
class LogProcessor:
    """A log processor used to format log information collected from
    ``runner.message_hub.log_scalars``.

    ``LogProcessor`` instance is built by runner and will format
    ``runner.message_hub.log_scalars`` to ``tag`` and ``log_str``, which can
    directly used by ``LoggerHook`` and ``MMLogger``. Besides, the argument
    ``custom_cfg`` of constructor can control the statistics method of logs.

    Args:
        window_size (int): default smooth interval. Defaults to 10.
        by_epoch (bool): Whether to format logs with epoch stype. Defaults to
            True.
        custom_cfg (list[dict], optional): Contains multiple log config dict,
            in which key means the data source name of log and value means the
            statistic method and corresponding arguments used to count the
            data source. Defaults to None.

            - If custom_cfg is None, all logs will be formatted via default
              methods, such as smoothing loss by default window_size. If
              custom_cfg is defined as a list of config dict, for example:
              [dict(data_src='loss', method='mean', log_name='global_loss',
              window_size='global')]. It means the log item ``loss`` will be
              counted as global mean and additionally logged as ``global_loss``
              (defined by ``log_name``). If ``log_name`` is not defined in
              config dict, the original logged key will be overwritten.

            - The original log item cannot be overwritten twice. Here is
              an error example:
              [dict(data_src='loss', method='mean', window_size='global'),
              dict(data_src='loss', method='mean', window_size='epoch')].
              Both log config dict in custom_cfg do not have ``log_name`` key,
              which means the loss item will be overwritten twice.

            - For those statistic methods with the ``window_size`` argument,
              if ``by_epoch`` is set to False, ``windows_size`` should not be
              `epoch` to statistics log value by epoch.
        num_digits (int): The number of significant digit shown in the
            logging message. Defaults to 4.
        log_with_hierarchy (bool): Whether to log with hierarchy. If it is
            True, the information is written to visualizer backend such as
            :obj:`LocalVisBackend` and :obj:`TensorboardBackend`
            with hierarchy. For example, ``loss`` will be saved as
            ``train/loss``, and accuracy will be saved as ``val/accuracy``.
            Defaults to False.
            `New in version 0.7.0.`
        mean_pattern (str): This is a regular expression used to match the log
            that need to be included in the smoothing statistics.
            `New in version 0.7.3.`

    Examples:
        >>> # `log_name` is defined, `loss_large_window` will be an additional
        >>> # record.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       log_name='loss_large_window',
        >>>                       method_name='mean',
        >>>                       window_size=100)])
        >>> # `log_name` is not defined. `loss` will be overwritten.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100)])
        >>> # Record loss with different statistics methods.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       log_name='loss_large_window',
        >>>                       method_name='mean',
        >>>                       window_size=100),
        >>>                  dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100)])
        >>> # Overwrite loss item twice will raise an error.
        >>> log_processor = dict(
        >>>     window_size=10,
        >>>     by_epoch=True,
        >>>     custom_cfg=[dict(data_src='loss',
        >>>                       method_name='mean',
        >>>                       window_size=100),
        >>>                  dict(data_src='loss',
        >>>                       method_name='max',
        >>>                       window_size=100)])
        AssertionError
    """

    def __init__(self,
                 window_size=10,
                 by_epoch=True,
                 custom_cfg: Optional[List[dict]] = None,
                 num_digits: int = 4,
                 log_with_hierarchy: bool = False,
                 mean_pattern=r'.*(loss|time|data_time|grad_norm).*'):
        self.window_size = window_size
        self.by_epoch = by_epoch
        self.custom_cfg = custom_cfg if custom_cfg else []
        self.num_digits = num_digits
        self.log_with_hierarchy = log_with_hierarchy
        self.mean_pattern = re.compile(mean_pattern)
        self._check_custom_cfg()

    def get_log_after_iter(self, runner, batch_idx: int,
                           mode: str) -> Tuple[dict, str]:
        """Format log string after training, validation or testing iteration.

        Args:
            runner (Runner): The runner of training phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner, train, test or val.

        Return:
            Tuple[dict, str]: Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert mode in ['train', 'test', 'val']
        # Overwrite ``window_size`` defined in ``custom_cfg`` to int value.
        parsed_cfg = self._parse_windows_size(runner, batch_idx,
                                              self.custom_cfg)
        # log_tag is used to write log information to terminal
        log_tag = self._collect_scalars(parsed_cfg, runner, mode)
        if mode == 'val':
            val_loss_tag = self._collect_scalars(parsed_cfg, runner, 'val')
            for key, value in val_loss_tag.items():
                log_tag[f'val/{key}'] = value
        # If `self.log_with_hierarchy` is False, the tag is the same as
        # log_tag. Otherwise, each key in tag starts with prefix `train`,
        # `test` or `val`
        if not self.log_with_hierarchy:
            tag = copy.deepcopy(log_tag)
        else:
            tag = self._collect_scalars(parsed_cfg, runner, mode, True)

        # Record learning rate.
        lr_str_list = []
        for key, value in tag.items():
            if key.endswith('lr'):
                key = self._remove_prefix(key, f'{mode}/')
                log_tag.pop(key)
                lr_str_list.append(f'{key}: '
                                   f'{value:.{self.num_digits}e}')
        lr_str = ' '.join(lr_str_list)
        # Format log header.
        # by_epoch == True
        #   train/val: Epoch [5][5/10]  ...
        #   test: Epoch [5/10]
        # by_epoch == False
        #  train: Epoch [5/10000] ... (divided by `max_iter`)
        #  val/test: Epoch [5/2000] ... (divided by length of dataloader)
        if self.by_epoch:
            # Align the iteration log:
            # Epoch(train)  [  9][010/270]
            # ...                 ||| |||
            # Epoch(train)  [ 10][100/270]
            dataloader_len = self._get_dataloader_size(runner, mode)
            cur_iter = self._get_iter(runner, batch_idx)
            cur_iter_str = str(cur_iter).rjust(len(str(dataloader_len)))
            if mode in ['train', 'val']:
                cur_epoch = self._get_epoch(runner, mode)
                if not (isinstance(runner._train_loop, dict)
                        or runner._train_loop is None):
                    # Right Align the epoch log:
                    # Epoch(train)   [9][100/270]
                    # ...             ||
                    # Epoch(train) [100][100/270]
                    max_epochs = runner.max_epochs
                    # 3 means the three characters: "[", "]", and " " occupied
                    # in " [{max_epochs}]"
                    cur_epoch_str = f'[{cur_epoch}]'.rjust(
                        len(str(max_epochs)) + 3, ' ')
                else:
                    cur_epoch_str = f'[{cur_epoch}]'
                tag['epoch'] = cur_epoch
                log_str = (f'Epoch({mode}){cur_epoch_str}'
                           f'[{cur_iter_str}/{dataloader_len}]  ')
            else:
                log_str = (f'Epoch({mode}) '
                           f'[{cur_iter_str}/{dataloader_len}]  ')
        else:
            if mode == 'train':
                cur_iter = self._get_iter(runner, batch_idx)
                cur_iter_str = str(cur_iter).rjust(len(str(runner.max_iters)))
                log_str = (f'Iter({mode}) '
                           f'[{cur_iter_str}/{runner.max_iters}]  ')
            else:
                dataloader_len = self._get_dataloader_size(runner, mode)
                cur_iter_str = str(batch_idx + 1).rjust(
                    len(str(dataloader_len)))
                log_str = (f'Iter({mode}) [{cur_iter_str}/{dataloader_len}]  ')
        # Add global iter.
        if isinstance(runner._train_loop, dict) or runner._train_loop is None:
            tag['iter'] = 0
        else:
            tag['iter'] = runner.iter + 1
        # Concatenate lr, momentum string with log header.
        log_str += f'{lr_str}  '
        # If IterTimerHook used in runner, eta, time, and data_time should be
        # recorded.
        if (all(item in log_tag for item in ['time', 'data_time'])
                and 'eta' in runner.message_hub.runtime_info):
            eta = runner.message_hub.get_info('eta')
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            log_str += f'eta: {eta_str}  '
            log_str += (f'time: {log_tag["time"]:.{self.num_digits}f}  '
                        f'data_time: '
                        f'{log_tag["data_time"]:.{self.num_digits}f}  ')
            # Pop recorded keys
            log_tag.pop('time')
            log_tag.pop('data_time')

        # If cuda is available, the max memory occupied should be calculated.
        if is_cuda_available():
            max_memory = self._get_max_memory(runner)
            log_str += f'memory: {max_memory}  '
            tag['memory'] = max_memory
        # Loop left keys to fill `log_str`.
        if mode in ('train', 'val'):
            log_items = []
            for name, val in log_tag.items():
                if mode == 'val' and not name.startswith('val/loss'):
                    continue
                if isinstance(val, float):
                    val = f'{val:.{self.num_digits}f}'
                log_items.append(f'{name}: {val}')
            log_str += '  '.join(log_items)
        return tag, log_str

    

    def get_log_after_epoch(self, runner, batch_idx: int, mode: str, with_non_scalar: bool = False) -> Tuple[dict, str]:
        assert mode in ['test', 'val'], ('`_get_metric_log_str` only accept val or test mode, but got ' f'{mode}')
        dataloader_len = self._get_dataloader_size(runner, mode)
        
        if self.by_epoch:
            if mode == 'val':
                cur_epoch = self._get_epoch(runner, mode)
                log_str = (f'Epoch({mode}) [{cur_epoch}][{dataloader_len}/' f'{dataloader_len}]  ')
            else:
                log_str = (f'Epoch({mode}) [{dataloader_len}/{dataloader_len}]  ')
        else:
            log_str = (f'Iter({mode}) [{dataloader_len}/{dataloader_len}]  ')
        
        custom_cfg_copy = copy.deepcopy(self.custom_cfg)
        custom_keys = [
            self._remove_prefix(cfg['data_src'], f'{mode}/')
            for cfg in custom_cfg_copy
        ]
        
        if 'time' not in custom_keys:
            custom_cfg_copy.append(dict(data_src='time', window_size='epoch', method_name='mean'))
        if 'data_time' not in custom_keys:
            custom_cfg_copy.append(dict(data_src='data_time', window_size='epoch', method_name='mean'))
        
        parsed_cfg = self._parse_windows_size(runner, batch_idx, custom_cfg_copy)
        ori_tag = self._collect_scalars(parsed_cfg, runner, mode, self.log_with_hierarchy)
        
        non_scalar_tag = self._collect_non_scalars(runner, mode)
        
        tag = OrderedDict()
        time_tag = OrderedDict()
        
        for key, value in ori_tag.items():
            if key in (f'{mode}/time', f'{mode}/data_time', 'time', 'data_time'):
                time_tag[key] = value
            else:
                tag[key] = value
        
        log_items = []
        log_str += '  '
        
        for name, val in chain(tag.items(), non_scalar_tag.items(), time_tag.items()):
            if isinstance(val, float):
                val = f'{val:.{self.num_digits}f}'
            if isinstance(val, (torch.Tensor, np.ndarray)):
                val = f'\n{val}\n'
            log_items.append(f'{name}: {val}')
        
        log_str += '  '.join(log_items)

        if with_non_scalar:
            tag.update(non_scalar_tag)
        tag.update(time_tag)
        
        val_loss = ori_tag.get('loss', None) if mode == 'val' else None
        if val_loss is not None:
            log_str += f'  {mode}/loss: {val_loss:.{self.num_digits}f}'  # Include val loss
        
        return tag, log_str

    def _collect_scalars(self,
                         custom_cfg: List[dict],
                         runner,
                         mode: str,
                         reserve_prefix: bool = False) -> dict:
        """Collect log information to compose a dict according to mode.

        Args:
            custom_cfg (List[dict]): A copy of ``self.custom_cfg`` with int
                ``window_size``.
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.
            reserve_prefix (bool): Whether to reserve the prefix of the key.

        Returns:
            dict: Statistical values of logs.
        """
        custom_cfg = copy.deepcopy(custom_cfg)
        tag = OrderedDict()
        # history_scalars of train/val/test phase.
        history_scalars = runner.message_hub.log_scalars
        # corresponding mode history_scalars
        mode_history_scalars = OrderedDict()
        # extract log scalars and remove prefix to `mode_history_scalars`
        # according to mode.
        for prefix_key, log_buffer in history_scalars.items():
            if prefix_key.startswith(mode):
                if not reserve_prefix:
                    key = self._remove_prefix(prefix_key, f'{mode}/')
                else:
                    key = prefix_key
                mode_history_scalars[key] = log_buffer
        for key in mode_history_scalars:
            # Update the latest learning rate and smoothed time logs.
            if re.search(self.mean_pattern, key) is not None:
                tag[key] = mode_history_scalars[key].mean(self.window_size)
            else:
                # Default statistic method is current.
                tag[key] = mode_history_scalars[key].current()
        # Update custom keys.
        for log_cfg in custom_cfg:
            data_src = log_cfg.pop('data_src')
            log_name = log_cfg.pop('log_name', data_src)
            if reserve_prefix:
                data_src = f'{mode}/{data_src}'
                log_name = f'{mode}/{log_name}'
            # log item in custom_cfg could only exist in train or val
            # mode.
            if data_src in mode_history_scalars:
                tag[log_name] = mode_history_scalars[data_src].statistics(
                    **log_cfg)
        return tag

    def _collect_non_scalars(self, runner, mode: str) -> dict:
        """Collect log information to compose a dict according to mode.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.

        Returns:
            dict: non-scalar infos of the specified mode.
        """
        # infos of train/val/test phase.
        infos = runner.message_hub.runtime_info
        # corresponding mode infos
        mode_infos = OrderedDict()
        # extract log info and remove prefix to `mode_infos` according to mode.
        for prefix_key, value in infos.items():
            if prefix_key.startswith(mode):
                if self.log_with_hierarchy:
                    key = prefix_key
                else:
                    key = self._remove_prefix(prefix_key, f'{mode}/')
                mode_infos[key] = value
        return mode_infos

    def _remove_prefix(self, string: str, prefix: str):
        """Remove the prefix ``train``, ``val`` and ``test`` of the key."""
        if string.startswith(prefix):
            return string[len(prefix):]
        else:
            return string

    def _check_custom_cfg(self) -> None:
        """Check the legality of ``self.custom_cfg``."""

        def _check_window_size():
            for log_cfg in self.custom_cfg:
                if not self.by_epoch:
                    assert log_cfg['window_size'] != 'epoch', \
                        'window_size cannot be epoch if LoggerHook.by_epoch' \
                        ' is False.'

        def _check_repeated_log_name():
            # The `log_name` of the same data_src should not be repeated.
            # If `log_name` is not specified, `data_src` will be overwritten.
            # But only allowed to be overwritten once.
            check_set = set()
            for log_cfg in self.custom_cfg:
                assert 'data_src' in log_cfg
                data_src = log_cfg['data_src']
                log_name = log_cfg.get('log_name', data_src)
                assert log_name not in check_set, (
                    f'Found duplicate {log_name} for {data_src}. Please check'
                    'your `custom_cfg` for `log_processor`. You should '
                    f'neither define duplicate `{log_name}` for {data_src} '
                    f'nor do not define any {log_name} for multiple '
                    f'{data_src}, See more information in the docstring of '
                    'LogProcessor')

                check_set.add(log_name)

        _check_repeated_log_name()
        _check_window_size()

    def _parse_windows_size(self,
                            runner,
                            batch_idx: int,
                            custom_cfg: Optional[list] = None) -> list:
        """Parse window_size defined in custom_cfg to int value.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            batch_idx (int): The iteration index of current dataloader.
            custom_cfg (list): A copy of ``self.custom_cfg``. Defaults to None
                to keep backward compatibility.
        """
        if custom_cfg is None:
            custom_cfg = copy.deepcopy(self.custom_cfg)
        else:
            custom_cfg = copy.deepcopy(custom_cfg)
        for log_cfg in custom_cfg:
            window_size = log_cfg.get('window_size', None)
            if window_size is None or isinstance(window_size, int):
                continue
            elif window_size == 'epoch':
                log_cfg['window_size'] = batch_idx + 1
            elif window_size == 'global':
                log_cfg['window_size'] = runner.iter + 1
            else:
                raise TypeError(
                    'window_size should be int, epoch or global, but got '
                    f'invalid {window_size}')
        return custom_cfg

    def _get_max_memory(self, runner) -> int:
        """Returns the maximum GPU memory occupied by tensors in megabytes (MB)
        for a given device.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.

        Returns:
            The maximum GPU memory occupied by tensors in megabytes for a given
            device.
        """

        device = getattr(runner.model, 'output_device', None)
        return get_max_cuda_memory(device)

    def _get_iter(self, runner, batch_idx: int) -> int:
        """Get current iteration index.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            batch_idx (int): The iteration index of current
                dataloader. Defaults to None.

        Returns:
            int: The current global iter or inner iter.
        """
        if self.by_epoch:
            current_iter = batch_idx + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    def _get_epoch(self, runner, mode: str) -> int:
        """Get current epoch according to mode.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.

        Returns:
            int: The current epoch.
        """
        if mode == 'train':
            epoch = runner.epoch + 1
        elif mode == 'val':
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                epoch = 0
            else:
                # normal val mode
                # runner.epoch += 1 has been done before validation
                epoch = runner.epoch
        else:
            raise ValueError(
                f"runner mode should be 'train' or 'val', but got {mode}")
        return epoch

    def _get_cur_loop(self, runner, mode: str):
        """Get current loop according to mode.

        Args:
            runner (Runner): The runner of the training/validation/testing
                process.
            mode (str): Current mode of runner.

        Returns:
            BaseLoop: Current loop of runner.
        """
        # returns type hint will occur circular import
        if mode == 'train':
            return runner.train_loop
        elif mode == 'val':
            return runner.val_loop
        else:
            return runner.test_loop

    def _get_dataloader_size(self, runner, mode) -> int:
        """Get dataloader size of current loop.

        Args:
            runner (Runner): The runner of the training/validation/testing
            mode (str): Current mode of runner.

        Returns:
            int: The dataloader size of current loop.
        """
        return len(self._get_cur_loop(runner=runner, mode=mode).dataloader)

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