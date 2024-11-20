# src/data/processors/image_processor.py

import torch
from torchvision import transforms
from PIL import Image

class ImageProcessor:
    def __init__(self, resize_dim=(224, 224)):
        #  image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def process_image(self, image_path):
        """
        loads and processes an image from the given path.
        Returns a tensor of the processed image.
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"error processing image: {e}")
            return None

# Example usage
if __name__ == "__main__":
    processor = ImageProcessor()
    processed_image = processor.process_image("test_image.jpg")
    if processed_image is not None:
        print(f"processed image shape: {processed_image.shape}")