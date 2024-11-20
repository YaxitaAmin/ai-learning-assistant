# Multimodal Learning Engine

A comprehensive machine learning engine that combines mathematical problem generation and computer vision capabilities into a unified system. This project showcases the power of multimodal learning through interactive math problem generation, visual problem solving, and image processing.

## 🚀 Features

- **Math Problem Generation**
  - Generates problems in algebra, geometry, and calculus
  - Provides solutions with step-by-step explanations
  - Supports different difficulty levels

- **Vision Processing**
  - Image feature extraction using pretrained models
  - Support for image processing tasks
  - Feature encoding for further analysis

- **Visualization**
  - Geometric diagrams for math problems
  - Plots for mathematical concepts
  - Visual representations of solutions

## 📋 Prerequisites

- Python 3.x
- GPU support (optional, for faster processing)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-learning-engine.git
cd multimodal-learning-engine
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

```
torch
numpy
matplotlib
Pillow
```

## 🎯 Usage

### Basic Usage

Run the main script:
```bash
python main.py
```

This will:
1. Generate math problems across different topics
2. Create visualizations for geometry problems
3. Demonstrate image encoding capabilities

### Example Outputs

#### Math Problem Generation
```
=== Problem Set ===
Difficulty: 0.2
Topic: algebra
Problem: Solve for x: 7x + 6 = 34
Solution: 4
Explanation:
1. Subtract 6 from both sides
2. Divide both sides by 7
3. x = 4
```

#### Vision Processing
```python
# Process an image using the vision encoder
from engine import VisionEncoder

encoder = VisionEncoder()
embedding = encoder.process_image("your_image.jpg")
```

## 📊 Generated Visualizations

The engine automatically generates visualizations for:
- Geometry problems (triangles, circles)
- Mathematical plots
- Problem-specific diagrams

All visualizations are saved as .png files in the output directory.

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Your Name - [yaxita2003@gmail.com](mailto:yaxita2003@gmail.com)

Project Link: [https://github.com/YaxitaAmin/multimodal-learning-engine](https://github.com/YaxitaAmin/multimodal-learning-engine)
