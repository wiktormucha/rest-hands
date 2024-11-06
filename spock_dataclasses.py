from typing import List
from spock import spock
from spock import SpockBuilder
from models import models
from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple
import torch.optim as optim

# Make enum class


class Project(Enum):
    Rehab_Action_Recognition = 'Rehab_Action_Recognition'
    Exercise_Form_Evaluation = 'Exercise_Form_Evaluation'
    Exercise_Repetition_Counting = 'Exercise_Repetition_Counting'
    Pick_Detection = 'Pick_Detection'


class ModelType(Enum):
    SwinS = 'SwinS'
    SlowFast = 'SlowFast'
    MViT_V2s = 'MViT_V2s'
    EfficientNet = 'EfficientNet'


class TaskType(Enum):
    action_correctness = 'action_correctness'
    exercise_recognition = 'exercise_recognition'
    repetition_counting = 'repetition_counting'
    pick_detection = 'pick_detection'


@spock
class RehabCfg():
    dataset_root: bool = '/data/wmucha/datasets/rehab_dataset/Dataset'
    seed_num: int = 42
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 0.001
    max_epochs: int = 10
    device: int = 0
    model_type: ModelType
    load_checkpoint: bool = False
    optimizer_type: str = 'SGD'
    weight_decay: float = 0.0
    early_stopping: int = 3
    checkpoint_name: str = 'checkpoints/weights.pth'
    cross_subject: bool = False
    subject: str = 'Subject_2'
    img_size: List[int] = [256, 256]
    use_scheduler: bool = False
    scheduler_milestones: List[int] = [5, 8]
    task: TaskType
    num_frames: int = 32
    project_name: Project
    bacbone_freeze: bool = False
