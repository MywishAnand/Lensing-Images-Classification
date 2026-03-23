import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class LensClassifier(nn.Module):
    def __init__(self, num_classes=3, in_channels=1):
        super(LensClassifier, self).__init__()
        
        # Use pretrained weights to speed up convergence
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the first conv layer if we're not using 3 channels
        if in_channels != 3:
            original_weights = self.model.conv1.weight.data
            self.model.conv1 = nn.Conv2d(
                in_channels, 
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size, 
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding, 
                bias=self.model.conv1.bias is not None
            )
            # Initialize the new 1-channel weights with the mean of the original 3 channels
            if in_channels == 1:
                self.model.conv1.weight.data = original_weights.mean(dim=1, keepdim=True)
            
        # Adapt final fully connected layer to output num_classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = LensClassifier(num_classes=3, in_channels=1)
    dummy_input = torch.randn(8, 1, 150, 150)
    output = model(dummy_input)
    print("Model initialized successfully.")
    print("Dummy input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
