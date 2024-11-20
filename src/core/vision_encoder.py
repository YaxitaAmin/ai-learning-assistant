# src/core/vision_encoder.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(VisionEncoder, self).__init__()
        
        # Convolutional layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adaptive pooling to ensure fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer to produce final embedding
        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def forward(self, x):
        # If input is a PIL Image, convert to tensor
        if isinstance(x, Image.Image):
            x = self.transform(x).unsqueeze(0)
        
        # Extract features
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Generate embedding
        embedding = self.fc(x)
        return embedding
    
    def encode_image(self, image_path):
        """
        Encode an image from file path
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return self(image)
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

# Test function to demonstrate usage
def test_vision_encoder():
    # Create encoder
    encoder = VisionEncoder()
    
    # Create a dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    
    # Encode image
    embedding = encoder(dummy_image)
    print("Embedding shape:", embedding.shape)
    return embedding

# # Uncomment to test
# if __name__ == "__main__":
#     test_vision_encoder()