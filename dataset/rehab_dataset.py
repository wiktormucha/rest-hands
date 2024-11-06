import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import random
from PIL import Image
import albumentations as A
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Create custom pytorch dataset class
action_classes_dict = {'1_coin_left': 0, '2_coin_right': 1, '3_pen_spin_left': 2, '4_pen_spin_right': 3, '5_pen_slide_left': 4, '6_pen_slide_right': 5, '7_ball_squeeze_left': 6, '8_ball_squeeze_right': 7, '9_bottle_squeeze_left': 8, '10_bottle_squeeze_right': 9, '11_bottle_curls_wrist_up_left': 10, '12_bottle_curls_wrist_up_right': 11,
                       '13_bottle_curls_wrist_down_left': 12, '14_bottle_curls_wrist_down_right': 13, '15_towel_left': 14, '16_towel_right': 15, '17_wrist_strech_left': 16, '18_wrist_strech_right': 17, '19_hand_slide_left': 18, '20_hand_slide_right': 19, '21_fist_left': 20, '22_fist_right': 21, '23_hand_flip_left': 22, '24_hand_flip_right': 23, '25_side_to_side': 24}

NUM_CLASSES = 25
PICK_DETECTION = False


MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]


def sample(input_frames: list, no_of_outframes, sampling_type: str):

    if len(input_frames) >= no_of_outframes:
        indxs_to_sample = np.arange(len(input_frames))

        if sampling_type == "uniform":

            # Uniformly susample the frames to match the no_of_outframes
            indxs_to_sample = np.linspace(
                0, len(input_frames)-1, no_of_outframes, dtype=int)

          #   indxs_to_sample = T.uniform_temporal_subsample(
          #       torch.tensor(indxs_to_sample), no_of_outframes, 0).tolist()

        elif sampling_type == "random":

            # randomly susample the frames to match the no_of_outframes
            indxs_to_sample = list(range(len(input_frames)))
            indxs_to_sample = random.sample(
                indxs_to_sample, no_of_outframes)
            indxs_to_sample.sort()

    else:
        indxs_to_sample = np.trunc(
            np.arange(0, no_of_outframes) * len(input_frames)/no_of_outframes).astype(int)

    return indxs_to_sample


class RehabHands(torch.utils.data.Dataset):
    def __init__(self, cfg, subset_type='train', transform=None):

        dataset_root = cfg.dataset_root
        self.dataset_root = dataset_root

        if subset_type == 'train':
            # self.sampling_type = 'random'
            self.sampling_type = 'uniform'
        else:
            self.sampling_type = 'uniform'

        self.transform = transform
        self.num_frames = cfg.num_frames
        self.model_type = cfg.model_type

        if cfg.task == 'exercise_recognition':
            if cfg.cross_subject:
                pth = os.path.join(dataset_root, 'labels',
                                   'action_labels_final_shuffled.csv')
                action_csv = pd.read_csv(
                    pth, sep=';')
            else:
                if subset_type == 'train':
                    pth = os.path.join(dataset_root, 'labels', 'train.csv')
                elif subset_type == 'val':
                    pth = os.path.join(
                        dataset_root, 'labels', 'validation.csv')
                else:
                    pth = os.path.join(dataset_root, 'labels', 'test.csv')

                action_csv = pd.read_csv(
                    pth, sep=',')
        elif cfg.task == 'action_correctness':
            if cfg.cross_subject:
                pth = os.path.join(
                    dataset_root, 'labels', 'action_labels_final_shuffled_actions_correctness.csv')
                action_csv = pd.read_csv(pth, sep=';')
            else:
                if subset_type == 'train':
                    pth = os.path.join(dataset_root, 'labels',
                                       'train_ex_corretness.csv')
                elif subset_type == 'val':
                    pth = os.path.join(dataset_root, 'labels',
                                       'validation_ex_corretness.csv')
                else:
                    pth = os.path.join(dataset_root, 'labels',
                                       'test_ex_corretness.csv')

                action_csv = pd.read_csv(
                    pth, sep=',')

        elif cfg.task == 'repetition_counting':
            if cfg.cross_subject:

                pth1 = os.path.join(dataset_root, 'labels',
                                    'train_repetitions.csv')
                pth2 = os.path.join(dataset_root, 'labels',
                                    'validation_repetitions.csv')
                pth3 = os.path.join(dataset_root, 'labels',
                                    'test_repetitions.csv')

                action_csv1 = pd.read_csv(
                    pth1, sep=',')
                action_csv2 = pd.read_csv(
                    pth2, sep=',')
                action_csv3 = pd.read_csv(
                    pth3, sep=',')
                action_csv = pd.concat(
                    [action_csv1, action_csv2, action_csv3], ignore_index=True)

            else:
                if subset_type == 'train':
                    pth = os.path.join(dataset_root, 'labels',
                                       'train_repetitions.csv')
                elif subset_type == 'val':
                    pth = os.path.join(dataset_root, 'labels',
                                       'validation_repetitions.csv')
                else:
                    pth = os.path.join(dataset_root, 'labels',
                                       'test_repetitions.csv')

                action_csv = pd.read_csv(
                    pth, sep=',')
        elif cfg.task == 'pick_detection':
            if cfg.cross_subject:

                pth1 = os.path.join(dataset_root, 'labels',
                                    'train_paths_pick.csv')
                pth2 = os.path.join(dataset_root, 'labels',
                                    'val_paths_pick.csv')
                pth3 = os.path.join(dataset_root, 'labels',
                                    'test_paths_pick.csv')

                action_csv1 = pd.read_csv(
                    pth1, sep=',')
                action_csv2 = pd.read_csv(
                    pth2, sep=',')
                action_csv3 = pd.read_csv(
                    pth3, sep=',')

                action_csv = pd.concat(
                    [action_csv1, action_csv2, action_csv3], ignore_index=True)
            else:
                if subset_type == 'train':
                    pth = os.path.join(dataset_root, 'labels',
                                       'train_paths_pick.csv')
                elif subset_type == 'val':
                    pth = os.path.join(dataset_root, 'labels',
                                       'val_paths_pick.csv')
                else:
                    pth = os.path.join(dataset_root, 'labels',
                                       'test_paths_pick.csv')

                action_csv = pd.read_csv(
                    pth, sep=',')
        else:
            raise ValueError(
                f'Task type not supported. Choose from: exercise_recognition, action_correctness, repetition_counting - {cfg.task}')

        if cfg.model_type == 'SlowFast' or cfg.model_type == 'MViT_V2s':
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]

        elif cfg.model_type == 'SwinS' or cfg.model_type == 'EfficientNet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.pytorch_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.subset_type = subset_type
        # Read each column of the csv file to seperate list files. THe columns are 'action_id': action_id_lst,'subject':subject_lst, 'clip_name':clip_name_lst, 'action_class':action_class_lst, 'action_correct': action_correct_lst, 'action_path':action_path_lst, 'action_start':action_start_lst, 'action_end':action_end_lst
        if cfg.task != 'pick_detection':
            self.action_id_lst = action_csv['action_id']
            self.subject_lst = action_csv['subject']
            self.clip_name_lst = action_csv['clip_name']
            self.action_class_lst = action_csv['action_class']
            if cfg.task == 'repetition_counting':
                self.action_correct_lst = [
                    0 for i in range(len(self.action_id_lst))]
            self.action_path_lst = action_csv['action_path']
            self.action_start_lst = action_csv['action_start']
            self.action_end_lst = action_csv['action_end']

            # # loop over iems i nthe list
            self.data = []
            for i in range(len(self.action_id_lst)):
                self.data.append({
                    'action_id': self.action_id_lst[i],
                    'subject': self.subject_lst[i],
                    'clip_name': self.clip_name_lst[i],
                    'action_class': self.action_class_lst[i],
                    'action_path': self.action_path_lst[i],
                    'action_start': self.action_start_lst[i],
                    'action_end': self.action_end_lst[i],
                })

            if cfg.cross_subject:
                if self.subset_type == 'test':
                    # Keep only items that habe subject equal to Subject_2
                    self.data = [
                        item for item in self.data if item['subject'] == cfg.subject]

                else:
                    # Keep all items except those that have subject equal to Subject_2
                    self.data = [
                        item for item in self.data if item['subject'] != cfg.subject]

                    if self.subset_type == 'train':
                        # Keep only first 90% of the items in self.data
                        self.data = self.data[:int(0.9*len(self.data))]
                    elif self.subset_type == 'val':
                        # Keep only last 10% of the items in self.data
                        self.data = self.data[int(0.9*len(self.data)):]

        elif cfg.task == 'pick_detection':
            self.paths = action_csv['path']
            self.labels = action_csv['label']

            self.data = []
            for i in range(len(self.paths)):
                self.data.append({
                    'img_pth': self.paths[i],
                    'label': self.labels[i]
                })

            if cfg.cross_subject:
                if self.subset_type == 'test':
                    # Keep only items that habe subject equal to Subject_2
                    self.data = [
                        item for item in self.data if cfg.subject in item['img_pth']]

                else:
                    # Keep all items except those that have subject equal to Subject_2
                    self.data = [
                        item for item in self.data if cfg.subject not in item['img_pth']]

                    if self.subset_type == 'train':
                        # Keep only first 90% of the items in self.data
                        self.data = self.data[:int(0.9*len(self.data))]
                    elif self.subset_type == 'val':
                        # Keep only last 10% of the items in self.data
                        self.data = self.data[int(0.9*len(self.data)):]

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the sample data.
        """

        if PICK_DETECTION:
            # For pick detection task
            img_pth = self.data[idx]['img_pth']
            label = self.data[idx]['label']

            # Load the image
            img = np.array(Image.open(img_pth))

            # Apply transformations
            transformed = self.transform(image=img)
            img = transformed['image']
            img = self.pytorch_transforms(img)

            return {
                'img_pth': img_pth,
                'action_tensor': img,
                'pick_label': label
            }
        else:
            # For other tasks
            action_id = self.data[idx]['action_id']
            subject = self.data[idx]['subject']
            clip_name = self.data[idx]['clip_name']
            action_class = action_classes_dict[self.data[idx]['action_class']]
            action_correct = self.data[idx]['action_correct']
            action_path = self.data[idx]['action_path']
            action_start = self.data[idx]['action_start']
            action_end = self.data[idx]['action_end']
            repetition_count = self.data[idx]['repetition_count']

            # Ensure action_correct is either 0 or 1
            assert action_correct in [
                0, 1], f'Action correctness should be 0 or 1 - {action_correct}'

            # Load frames of the video clip
            action_tensor, all_paths = self.get_action_tensor(
                pth=action_path, start=action_start, end=action_end, transform=self.transform)

            # Permute the tensor dimensions
            action_tensor = action_tensor.permute(1, 0, 2, 3)

            # Subsample the frames of action_tensor uniformly to match 8 frames
            if self.model_type == 'SlowFast':
                action_tensor_fast = action_tensor[:, ::4, :, :]
                action_tensor = {
                    'slow': action_tensor,
                    'fast': action_tensor_fast
                }

            return {
                'action_id': action_id,
                'subject': subject,
                'clip_name': clip_name,
                'action_label': action_class,
                'action_correct': action_correct,
                'action_path': action_path,
                'action_start': action_start,
                'action_end': action_end,
                'action_tensor': action_tensor,
                'repetition_count': repetition_count,
                'all_paths': all_paths,
            }

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def get_action_tensor(self, pth, start, end, transform):
        """
        Loads and transforms a sequence of images from a given path, start, and end indices.

        Args:
            pth (str): Path to the directory containing the images.
            start (int): Starting index of the images to load.
            end (int): Ending index of the images to load.
            transform (A.ReplayCompose): Albumentations transformation to apply.

        Returns:
            torch.Tensor: A tensor containing the transformed images.
            list: List of paths to the images.
        """

        # Adjust start and end indices if necessary
        if start == 0:
            start = 1
        if end == 901:
            end = 900

        transform_replay = None

        # Create a list of paths to load the images from the clip
        paths = [
            f'{self.dataset_root}/{pth}/{str(i).zfill(6)}.jpg' for i in range(start, end)
        ]

        # Sample indices to match the number of frames required
        indxs_to_sample = sample(paths, self.num_frames, 'uniform')

        # Load the images from the paths and stack them to create a tensor
        img_list = []
        for idx, i in enumerate(indxs_to_sample):

            # Check if the file exists
            if i >= len(paths):
                raise ValueError(
                    f'Index out of range: {i}, start: {start}, end: {end}, paths: {paths}'
                )

            if not os.path.exists(paths[i]):
                raise FileExistsError(
                    f'File does not exist: {paths[i]}, start: {start}, end: {end}')

            # Load the image
            image = np.array(Image.open(paths[i]))

            # Apply transformations
            transformed = albumentation_to_sequence(
                image=image, albumentations=transform, transform_replay=transform_replay, frame_idx=idx
            )
            image = transformed['image']
            image = self.pytorch_transforms(image)

            # Save the replayed transformation for subsequent frames
            if idx == 0:
                transform_replay = transformed['replay']

            img_list.append(image)

        # Stack the list of images to create a tensor
        action_tensor = torch.stack(img_list)

        return action_tensor, paths


def get_rehab_dataloader(config):
    """
    Creates and returns dataloaders for training, validation, and testing datasets for the RehabHands dataset.

    Args:
        config (object): Configuration object containing parameters such as image size, batch size, and number of workers.

    Returns:
        dict: A dictionary containing the following keys:
            - 'dataloader_train': DataLoader for the training dataset.
            - 'dataloader_val': DataLoader for the validation dataset.
            - 'dataloader_test': DataLoader for the testing dataset.

    The function performs the following steps:
    1. Defines data augmentation and transformation pipelines for training and validation datasets using Albumentations.
    2. Creates instances of the RehabHands dataset for training, validation, and testing subsets.
    3. Prints the lengths of the training, validation, and testing datasets.
    4. Creates DataLoader instances for the training, validation, and testing datasets.
    5. Returns a dictionary containing the DataLoader instances.
    """

    transform_train = A.ReplayCompose(
        [
            A.Rotate(always_apply=True, p=0.2, limit=(-45, 45), interpolation=0, border_mode=4,
                     value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False),
            A.OneOf([
                A.ToSepia(always_apply=False, p=1.0),
                A.ToGray(always_apply=False, p=1.0),
                A.Superpixels(always_apply=False, p=1.0, p_replace=(
                    0.13, 0.13), n_segments=(521, 521), max_size=220, interpolation=0),
                A.Sharpen(always_apply=False, p=1.0, alpha=(
                    0.2, 0.31), lightness=(0.5, 1.0)),
                A.RandomToneCurve(always_apply=False, p=1.0, scale=0.48),
                A.RandomGamma(always_apply=False, p=1.0,
                              gamma_limit=(24, 200), eps=None),
                A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(
                    -0.49, 0.45), contrast_limit=(-0.53, 0.45), brightness_by_max=True),
                A.RGBShift(always_apply=False, p=1.0, r_shift_limit=(-12, 22),
                           g_shift_limit=(-13, 20), b_shift_limit=(-20, 20)),
                A.ChannelShuffle(always_apply=False, p=1.0),
                A.ColorJitter(always_apply=False, p=1.0, brightness=(0.8, 1.2), contrast=(
                    0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
                A.Downscale(always_apply=False, p=1.0,
                            scale_min=0.5, scale_max=0.99),
                A.MotionBlur(always_apply=False, p=1.0,
                             blur_limit=(5, 9), allow_shifted=True),
            ], p=0.8),
            A.OneOf([
                A.Resize(
                    config.img_size[0], config.img_size[1]),
                A.RandomResizedCrop(always_apply=True, p=0.5, height=config.img_size[0],
                                    width=config.img_size[1], scale=(0.7, 1.0), ratio=(1, 1), interpolation=0),
                A.Compose([
                    A.Rotate(always_apply=True, p=0.5, limit=(-30, 30), interpolation=0, border_mode=4,
                             value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False),
                    A.RandomResizedCrop(always_apply=True, p=0.5, height=config.img_size[0],
                                        width=config.img_size[1], scale=(0.7, 1.0), ratio=(1, 1), interpolation=0),
                ]),
            ], p=1.0),

        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    transform_val = A.ReplayCompose([
        A.Resize(
            config.img_size[0], config.img_size[1]),
    ])

    dataset_train = RehabHands(
        cfg=config, subset_type='train', transform=transform_train)
    dataset_val = RehabHands(
        cfg=config, subset_type='val', transform=transform_val)
    dataset_test = RehabHands(
        cfg=config, subset_type='test', transform=transform_val)

    len_train = len(dataset_train)
    len_val = len(dataset_val)
    len_test = len(dataset_test)

    print(f'Length of train dataset: {len_train}')
    print(f'Length of val dataset: {len_val}')
    print(f'Length of test dataset: {len_test}')

    # Sum and print
    print(f'Total number of samples: {len_train + len_val + len_test}')

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return {
        'dataloader_train': dataloader_train,
        'dataloader_val': dataloader_val,
        'dataloader_test': dataloader_test
    }


def albumentation_to_sequence(image, albumentations, transform_replay, frame_idx):
    """
    Apply albumentation transformations to a sequence of images.

    Parameters:
    image (numpy array): The input image.
    albumentations (A.ReplayCompose): The albumentation transformations to apply.
    transform_replay (dict): The replayed transformations to apply to subsequent frames.
    frame_idx (int): The index of the current frame in the sequence.

    Returns:
    dict: A dictionary containing the transformed image and the replayed transformations.
    """
    ret = {}
    # Apply for the first frame a new unique transformation
    if frame_idx == 0:
        transformed = albumentations(image=image)
        ret['replay'] = transformed["replay"]
    # For rest of frames apply replayed transformation
    else:
        transformed = A.ReplayCompose.replay(
            saved_augmentations=transform_replay, image=image)

    ret['image'] = transformed['image']
    return ret
