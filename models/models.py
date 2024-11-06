import torch
import torch.nn as nn
import torchvision
from functools import partial
from dataset.rehab_dataset import NUM_CLASSES
from torchvision.models import EfficientNet_V2_S_Weights


class SwinS(nn.Module):
    def __init__(self, out_classes=NUM_CLASSES):
        super(SwinS, self).__init__()
        self.model = torchvision.models.video.swin3d_s(
            weights='KINETICS400_V1')
        self.model.head = nn.Sequential(torch.nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(in_features=768, out_features=out_classes, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x

    def replace_last_layer(self, new_out_classes):
        # Access the last layer of self.linear, which is a nn.Linear layer
        last_layer = self.model.head[-1]
        # Create a new nn.Linear layer with the same in_features and new out_features
        new_last_layer = nn.Linear(
            in_features=last_layer.in_features, out_features=new_out_classes)
        # Replace the last layer in the self.linear Sequential container
        self.model.head[-1] = new_last_layer


class SlowFast(nn.Module):
    def __init__(self, out_classes=NUM_CLASSES):
        super().__init__()
        model = torch.hub.load('facebookresearch/pytorchvideo',
                               'slowfast_r50', pretrained=True)
        self.backbone = nn.Sequential(*list(model.blocks.children())[:-1])
        self.out_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.linear = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(2304, out_classes)
        )

    def forward(self, inputs):

        x = [inputs['fast'].reshape(-1, 3, 8, 256, 256),
             inputs['slow'].reshape(-1, 3, 32, 256, 256)]
        x = self.backbone(x)
        x = self.out_pool(x)
        x = x.reshape(x.shape[0], -1)
        out = self.linear((x))
        return out

    def replace_last_layer(self, new_out_classes):
        # Access the last layer of self.linear, which is a nn.Linear layer
        last_layer = self.linear[-1]
        # Create a new nn.Linear layer with the same in_features and new out_features
        new_last_layer = nn.Linear(
            in_features=last_layer.in_features, out_features=new_out_classes)
        # Replace the last layer in the self.linear Sequential container
        self.linear[-1] = new_last_layer


class MViT_V2s(nn.Module):
    def __init__(self, out_classes=NUM_CLASSES):
        super(MViT_V2s, self).__init__()
        self.model = torchvision.models.video.mvit_v2_s(
            weights='KINETICS400_V1')
        self.model.head = nn.Sequential(torch.nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(in_features=768, out_features=out_classes, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x

    def replace_last_layer(self, new_out_classes):
        # Access the last layer of self.linear, which is a nn.Linear layer
        last_layer = self.model.head[-1]
        # Create a new nn.Linear layer with the same in_features and new out_features
        new_last_layer = nn.Linear(
            in_features=last_layer.in_features, out_features=new_out_classes)
        # Replace the last layer in the self.linear Sequential container
        self.model.head[-1] = new_last_layer


class EfficientNet(nn.Module):
    """
    Efficientnet backbone without last max pooling. Outputs dimmensions of Bx4x4x1280.
    """

    def __init__(self, out_classes: int):
        """
        Init
        """
        super().__init__()
        self.model = torchvision.models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights)

        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=out_classes, bias=True)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass

        Args:
             x (torch.tensor): Input tensor
        Returns:
             torch.tensor: Output tensor
        """

        return self.model(x)
