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