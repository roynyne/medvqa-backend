import torch
import torch.nn as nn
from torchvision import models


# Image Encoder (VGG16)
class ImageEncoder(nn.Module):
    def __init__(self, feat_dim=512):
        super(ImageEncoder, self).__init__()
        # # using vgg16 model
        # base = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # # Modifying the first convolutional layer for grayscale input
        # base.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # # Removing the classification layers
        # self.vision_encoder = nn.Sequential(*list(base.children())[:-1])
        
        # # Adaptive pooling and feature projection
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.img_feat = nn.Linear(512, feat_dim)

        ## Using Resnet18 model
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_encoder = base_model

        self.vision_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Reinitialize conv1 weights
        nn.init.kaiming_normal_(self.vision_encoder.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Remove fully connected layer (used for classification)
        in_features = self.vision_encoder.fc.in_features
        self.vision_encoder.fc = nn.Identity()  # Remove FC layer

        # Projection layer to reduce feature dimensions
        self.img_feat = nn.Linear(in_features, feat_dim)

    def forward(self, x):
        x = self.vision_encoder(x)
        # when using vgg16 baseline model
        # x = self.avgpool(x).view(x.size(0), -1)  # Flatten
        x = self.img_feat(x)
        return x  # Output shape: (batch_size, feat_dim)
    

if __name__ == '__main__':
    # test image encoder
    test_input = torch.randn(2, 1, 224, 224)
    model = ImageEncoder()
    output = model(test_input)
    print(model)    
    print(output.shape)

    