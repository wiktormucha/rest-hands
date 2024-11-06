from dataset.rehab_dataset import NUM_CLASSES
import numpy as np
import torch
from tqdm import tqdm
import wandb
import torchmetrics as metrics
import torch.nn as nn
import os

# DEBUG = True
DEBUG = False
MAX_BATCHES = 10
UNFREEZE_STEP = 5


def save_best_model(model, run_name, new_value: float, best_value, save_on_type: str):
    """
    Saves the best model based on the specified criteria.
    Args:
        model (torch.nn.Module): The model to save.
        run_name (str): The name of the run.
        new_value (float): The new value to compare.
        best_value (float): The current best value.
        save_on_type (str): The type of comparison ('greater_than' or 'less_than').
    Returns:
        float: The updated best value.
    """
    if not os.path.exists(f'checkpoints/{run_name}'):
        os.makedirs(f'checkpoints/{run_name}')

    best_value_ret = best_value
    if save_on_type == 'greater_than':
        if new_value >= best_value:
            best_value_ret = new_value
            print("Saving best model..")
            torch.save(model.state_dict(),
                       f'checkpoints/{run_name}/checkpoint_best.pth')
    else:
        if new_value <= best_value:
            best_value_ret = new_value
            print("Saving best model..")
            torch.save(model.state_dict(),
                       f'checkpoints/{run_name}/checkpoint_best.pth')

    return best_value_ret


class TrainerAR:
    """
    Class for training the model.
    """

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim, training_config, wandb_logger: wandb = None, scheduler: torch.optim = None) -> None:
        """
        Initialization.
        Args:
            model (torch.nn.Module): The model to train.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim): The optimizer.
            training_config (dict): Configuration dictionary.
            wandb_logger (wandb, optional): Weights and Biases logger. Defaults to None.
            scheduler (torch.optim, optional): Learning rate scheduler. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": [], "test": []}
        self.epochs = training_config.max_epochs
        self.device = training_config.device
        self.scheduler = scheduler
        self.early_stopping_epochs = training_config.early_stopping
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.best_val_loss = 100000
        self.best_acc_val = 0
        self.wandb_logger = wandb_logger
        self.acc_val = []
        self.acc_test = []
        self.softmax = nn.Softmax(dim=1)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.model_type = training_config.model_type
        self.task = training_config.task
        self.bacbone_freeze = training_config.bacbone_freeze

    def train(self, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader, test_dataloader) -> torch.nn.Module:
        """
        Training loop.
        Args:
            train_dataloader (torch.utils.data.DataLoader): Training dataloader.
            val_dataloader (torch.utils.data.DataLoader): Validation dataloader.
            test_dataloader (torch.utils.data.DataLoader): Test dataloader.
        Returns:
            torch.nn.Module: Trained model.
        """
        nan_count = 0

        if self.bacbone_freeze:
            if self.model_type == 'SlowFast':
                unfreeze_layers = 4
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
                freez_count = len(self.model.backbone) - 1
            elif self.model_type == 'SwinS':
                unfreeze_layers = 5
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.model.head.parameters():
                    param.requires_grad = True
                for param in self.model.model.avgpool.parameters():
                    param.requires_grad = True
                for param in self.model.model.norm.parameters():
                    param.requires_grad = True
                freez_count = 6
            elif self.model_type == 'MViT_V2s':
                unfreeze_layers = 13
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.model.head.parameters():
                    param.requires_grad = True
                freez_count = 15
            elif self.model_type == 'EfficientNet':
                unfreeze_layers = 4
                for param in self.model.model.features.parameters():
                    param.requires_grad = False
                freez_count = 7

        for epoch in range(self.epochs):
            if epoch % UNFREEZE_STEP == 0 and epoch != 0 and freez_count >= unfreeze_layers:
                print('Unfreezing layer: ', freez_count)
                if self.model_type == 'SlowFast':
                    for layer_idx, layer in enumerate(self.model.backbone):
                        if layer_idx == freez_count:
                            for param in layer.parameters():
                                param.requires_grad = True
                            freez_count -= 1
                elif self.model_type == 'SwinS':
                    for layer_idx, (name, module) in enumerate(self.model.model.features.named_children()):
                        if layer_idx == freez_count:
                            for param in module.parameters():
                                param.requires_grad = True
                            freez_count -= 1
                elif self.model_type == 'MViT_V2s':
                    if epoch <= UNFREEZE_STEP:
                        for param in self.model.model.norm.parameters():
                            param.requires_grad = True
                    else:
                        for layer_idx, (name, module) in enumerate(self.model.model.blocks.named_children()):
                            if layer_idx == freez_count:
                                for param in module.parameters():
                                    param.requires_grad = True
                                freez_count -= 1
                elif self.model_type == 'EfficientNet':
                    for layer_idx, (name, module) in enumerate(self.model.model.features.named_children()):
                        if layer_idx == freez_count:
                            for param in module.parameters():
                                param.requires_grad = True
                            freez_count -= 1

            print('Trainable parameters in backbone: ', sum(p.numel()
                  for p in self.model.parameters() if p.requires_grad))

            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            print("Epoch: {}/{}, Train Loss={}, Val Loss={}, Val Acc={}".format(
                epoch + 1, self.epochs, np.round(self.loss["train"][-1], 10), np.round(self.loss["val"][-1], 10), self.acc_val[-1]))

            if torch.isnan(torch.tensor(self.loss["val"][-1])):
                nan_count += 1
            else:
                nan_count = 0

            if nan_count > 3:
                print("Early Stopping because of nan values in loss")
                break

            if self.scheduler is not None:
                self.scheduler.step()

            self.__save_best_model(acc_val=np.round(self.acc_val[-1], 10))

            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(
                    np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0
            else:
                val_loss = np.round(np.mean(
                    self.loss["val"][-self.early_stopping_avg:]), self.early_stopping_precision)
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0

            if self.wandb_logger is not None:
                self.wandb_logger.log({"acc": self.acc_val[-1], "train_loss": np.round(
                    self.loss["train"][-1], 10), "val_loss": np.round(self.loss["val"][-1], 10), "best_acc_val": self.best_acc_val})

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        self.model.load_state_dict(torch.load(
            f'checkpoints/{self.wandb_logger.name}/checkpoint_best.pth'))
        test_acc = self.test_model(test_dataloader=test_dataloader)
        if self.wandb_logger is not None:
            self.wandb_logger.log({"test_acc": test_acc})

        return self.model

    def _epoch_train(self, dataloader: torch.utils.data.DataLoader):
        """
        Training step in epoch.
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader.
        """
        self.model.train()
        running_loss = []

        for i, data in enumerate(tqdm(dataloader, 0)):
            if self.model_type == 'SlowFast':
                inputs = {'fast': data["action_tensor"]['fast'].to(self.device).type(torch.cuda.FloatTensor),
                          'slow': data["action_tensor"]['slow'].to(self.device).type(torch.cuda.FloatTensor)}
            else:
                inputs = data['action_tensor'].to(
                    self.device).type(torch.cuda.FloatTensor)

            if self.task == 'action_correctness':
                labels = data['action_correct'].to(self.device)
            elif self.task == 'exercise_recognition':
                labels = data['action_label'].to(self.device)
            elif self.task == 'repetition_counting':
                labels = data['repetition_count'].to(self.device)
            elif self.task == 'pick_detection':
                labels = data['pick_label'].to(self.device)
            else:
                print('Task not supported')
                break

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            running_loss.append(loss.item() * labels.shape[0])

            if DEBUG and i > MAX_BATCHES:
                break

        epoch_loss = sum(running_loss) / len(dataloader.dataset)
        self.loss["train"].append(epoch_loss)

    def test_model(self, test_dataloader: torch.utils.data.DataLoader):
        """
        Test the model.
        Args:
            test_dataloader (torch.utils.data.DataLoader): Test dataloader.
        Returns:
            float: Test accuracy.
        """
        self.__evaluation(dataloader=test_dataloader,
                          loss_list=self.loss["test"], acc_list=self.acc_test)
        print('Acc: ', self.acc_test[-1], ' Loss: ', self.loss["test"][-1])

        if self.task == 'repetition_counting':
            total_diff = sum(self.diff_dict.values())
            e0 = (self.diff_dict[0]
                  if 0 in self.diff_dict else 0) / total_diff * 100
            e1 = (self.diff_dict[1]
                  if 1 in self.diff_dict else 0) / total_diff * 100
            e2 = (self.diff_dict[2]
                  if 2 in self.diff_dict else 0) / total_diff * 100

            sum_diff = 0
            for key in self.diff_dict.keys():
                if key > 2:
                    sum_diff += self.diff_dict[key]
            e_gt_2 = sum_diff / total_diff

            mae = 0
            for key in self.diff_dict.keys():
                mae += key * self.diff_dict[key]
            mae = round((mae / total_diff), 2)

            e0 = round(e0, 2)
            e1 = round(e1, 2)
            e2 = round(e2, 2)
            e_gt_2 = round(e_gt_2, 2)
            print(
                f'e=0: {e0}, e=1: {e1}, e=2: {e2}, e >= 2: {e_gt_2}, MAE: {mae}')

        return self.acc_test[-1]

    def __evaluation(self, dataloader: torch.utils.data.DataLoader, loss_list, acc_list):
        """
        Evaluation step.
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader.
            loss_list (list): List to store loss values.
            acc_list (list): List to store accuracy values.
        """
        self.model.eval()
        running_loss = []

        if self.task == 'action_correctness' or self.task == 'pick_detection':
            Accuracy = metrics.Accuracy(
                task="multiclass", num_classes=2).to(self.device)
        elif self.task == 'exercise_recognition':
            Accuracy = metrics.Accuracy(
                task="multiclass", num_classes=NUM_CLASSES).to(self.device)
        elif self.task == 'repetition_counting':
            Accuracy = metrics.Accuracy(
                task="multiclass", num_classes=20).to(self.device)

        acc_lst = []
        self.diff_dict = {}

        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, 0)):
                if self.model_type == 'SlowFast':
                    inputs = {'fast': data["action_tensor"]['fast'].to(self.device).type(torch.cuda.FloatTensor),
                              'slow': data["action_tensor"]['slow'].to(self.device).type(torch.cuda.FloatTensor)}
                else:
                    inputs = data['action_tensor'].to(
                        self.device).type(torch.cuda.FloatTensor)

                if self.task == 'action_correctness':
                    labels = data['action_correct'].to(self.device)
                elif self.task == 'exercise_recognition':
                    labels = data['action_label'].to(self.device)
                elif self.task == 'repetition_counting':
                    labels = data['repetition_count'].to(self.device)
                elif self.task == 'pick_detection':
                    labels = data['pick_label'].to(self.device)
                else:
                    print('Task not supported')
                    break

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item() * labels.shape[0])

                if self.task == 'repetition_counting':
                    prob = self.softmax(outputs)
                    _, pred_rep = torch.max(prob, 1)
                    diff = torch.abs(pred_rep - labels)
                    t = 0
                    a = 0
                    for d in diff:
                        d = int(d.item())
                        if d in self.diff_dict:
                            self.diff_dict[d] += 1
                        else:
                            self.diff_dict[d] = 1
                        if d == 0:
                            a += 1
                        t += 1
                    acc = a / t
                else:
                    acc = Accuracy(outputs, labels).cpu().numpy()

                acc_lst.append(acc * labels.shape[0])

                if DEBUG and i > MAX_BATCHES:
                    break

            epoch_loss = sum(running_loss) / len(dataloader.dataset)
            loss_list.append(epoch_loss)
            accuracy = sum(acc_lst) / len(dataloader.dataset)
            acc_list.append(accuracy)

    def _epoch_eval(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluation step in epoch.
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader.
        """
        self.__evaluation(dataloader=dataloader,
                          loss_list=self.loss["val"], acc_list=self.acc_val)

    def __save_best_model(self, acc_val: float):
        """
        Saves the best model based on validation accuracy.
        Args:
            acc_val (float): Current validation accuracy.
        """
        if self.wandb_logger is not None:
            if not os.path.exists(f'checkpoints/{self.wandb_logger.name}'):
                os.makedirs(f'checkpoints/{self.wandb_logger.name}')

            if acc_val >= self.best_acc_val:
                self.best_acc_val = acc_val
                print("Saving best model..")
                torch.save(self.model.state_dict(
                ), f'checkpoints/{self.wandb_logger.name}/checkpoint_best.pth')
