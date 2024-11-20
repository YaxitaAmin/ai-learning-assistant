#main.py
# main.py

import sys
import os
import yaml
import torch
from src.core.vision_encoder import VisionEncoder
from src.data.generators.math_generator import MathProblemGenerator
from src.data.processors.image_processor import ImageProcessor
from src.data.processors.text_processor import TextProcessor

def main():
    # Load configuration
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Default configuration if file not found
        config = {
            'vision_encoder': {'embedding_dim': 512},
            'num_concepts': 1000,
            'embedding_dim': 128
        }
    
    # Initialize components
    vision_encoder = VisionEncoder(
        embedding_dim=config['vision_encoder'].get('embedding_dim', 512)
    )
    
    image_processor = ImageProcessor()
    text_processor = TextProcessor()

    # Initialize problem generator
    problem_generator = MathProblemGenerator()

    # Generate sample problem
    difficulty_levels = [0.2, 0.5, 0.8]
    for difficulty in difficulty_levels:
        print("\n" + "="*50)
        problem_data = problem_generator.generate_problem('algebra', difficulty)
        
        print(f"Difficulty Level: {difficulty}")
        print(f"Topic: {problem_data['topic']}")
        print(f"Problem: {problem_data['problem']}")
        print(f"Solution: {problem_data['solution']}")
        print(f"Explanation: {problem_data['explanation']}")

    # Demonstrate vision encoder
    print("\n" + "="*50)
    print("Vision Encoder Demonstration:")
    try:
        # Create a random image for testing
        import numpy as np
        from PIL import Image
        
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        dummy_image.save('test_image.png')
        
        embedding = vision_encoder.encode_image('test_image.png')
        print("Image Embedding Shape:", embedding.shape)
    except Exception as e:
        print(f"Error in vision encoder demo: {e}")

    # Test text processor
    print("\n" + "="*50)
    print("Text Processor Demonstration:")
    try:
        text = "This is a sample sentence for processing."
        processed_text = text_processor.process_text(text)
        print(f"Processed text: {processed_text}")
    except Exception as e:
        print(f"Error in text processor demo: {e}")

if __name__ == "__main__":
    main()
