# from datasets.h2o import H2O_actions
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
# import torch.optim as optim
from spock_dataclasses import *
# import importlib.util
from spock import SpockBuilder
from utils.general_utils import freeze_seeds
from utils.trainer import TrainerAR
from utils.general_utils import define_optimizer
from models.models import SwinS
from dataset.rehab_dataset import get_rehab_dataloader
import sys
import yaml
import wandb
from models.models import SlowFast, MViT_V2s, EfficientNet
import os
from utils.trainer import DEBUG

# TEST_ONLY = True
TEST_ONLY = False


def main() -> None:
    """
    Main training loop
    """
    #  run this command in linux terminal: export CUDA_LAUNCH_BLOCKING=1
    #  to get more detailed error messages
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # Build config
    config = SpockBuilder(RehabCfg, desc='Quick start example').generate()

    freeze_seeds(seed_num=config.RehabCfg.seed_num)

    wandbcfg_pth = sys.argv[2]
    # opening a file
    with open(wandbcfg_pth, 'r') as stream:
        try:
            # Converts yaml document to python object
            wandbcfg = yaml.safe_load(stream)

        # Program to convert yaml file to dictionary
        except yaml.YAMLError as e:
            print(e)

    if DEBUG or TEST_ONLY:
        logger = None
    else:
        logger = wandb.init(
            # set the wandb project where this run will be logged
            project=config.RehabCfg.project_name,
            config=wandbcfg)

    dataloaders = get_rehab_dataloader(config.RehabCfg)
    exit()
    # Create model
    # Make model from the string passed in config.RehabCfg.model_name
    ModelClass = globals()[config.RehabCfg.model_type]

    if config.RehabCfg.task == 'action_correctness':
        model = ModelClass(out_classes=2)
    elif config.RehabCfg.task == 'exercise_recognition':
        model = ModelClass(out_classes=25)
    elif config.RehabCfg.task == 'pick_detection':
        model = ModelClass(out_classes=2)

    elif config.RehabCfg.task == 'repetition_counting':
        model = ModelClass(out_classes=21)
    else:
        raise ValueError('Task not recognized')

    # # Load state dict
    # wandb_run_name = 'noble-frog-103'

    if config.RehabCfg.load_checkpoint:
        model.load_state_dict(torch.load(
            f'checkpoints/{config.RehabCfg.checkpoint_name}/checkpoint_best.pth'))
        if config.RehabCfg.task == 'action_correctness':
            model.replace_last_layer(new_out_classes=2)
    model = model.to(config.RehabCfg.device)

   #     # If loading weights from checkpoin
   #     if config.ModelConfig.load_checkpoint:
   #         model.load_state_dict(torch.load(
   #             config.ModelConfig.checkpoint_path, map_location=torch.device(config.RehabCfg.device)))
   #         print("Model's checkpoint loaded")

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = define_optimizer(model, config.RehabCfg)

    if config.RehabCfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.RehabCfg.scheduler_milestones, gamma=0.5, last_epoch=- 1, verbose=True)
    else:
        scheduler = None

    trainer = TrainerAR(model, criterion, optimizer,
                        config.RehabCfg, scheduler=scheduler, wandb_logger=logger)
    print(f'Starting training on device: {config.RehabCfg.device}')

    if TEST_ONLY:
        trainer.test_model(test_dataloader=dataloaders['dataloader_val'])
        trainer.test_model(test_dataloader=dataloaders['dataloader_test'])
    else:
        trainer.train(train_dataloader=dataloaders['dataloader_train'],
                      val_dataloader=dataloaders['dataloader_val'],
                      test_dataloader=dataloaders['dataloader_test'])


if __name__ == '__main__':
    main()
