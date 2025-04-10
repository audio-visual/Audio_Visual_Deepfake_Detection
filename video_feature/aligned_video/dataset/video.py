import sys
sys.path.append('/home/ubuntu/sn15_share_dir/av_deepfake/LAV-DF')
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Callable, Any, Union, Tuple

import einops
import numpy as np
import scipy as sp
import torch
import torchaudio
from einops import rearrange
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.nn import functional as F, Identity
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset

from utils import read_json, read_video, padding_video, padding_audio, resize_video, iou_with_anchors, ioa_with_anchors


@dataclass
class Metadata:
    file: str
    n_fakes: int
    fake_periods: List[List[int]]
    duration: float
    original: Optional[str]
    modify_video: bool
    modify_audio: bool
    split: str
    video_frames: int
    audio_channels: int
    audio_frames: int


T_LABEL = Union[Tensor, Tuple[Tensor, Tensor, Tensor]]


