# 🎨 Data Augmentation CLI Tool

A powerful and user-friendly command-line interface for augmenting image datasets with various transformations. Perfect for machine learning projects, computer vision tasks, and expanding your training datasets.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)

## ✨ Features

- **🖼️ Multiple Image Formats**: Support for JPG, JPEG, PNG, BMP, TIFF, and WEBP
- **🔧 11 Transformation Types**: Comprehensive set of augmentation techniques
- **📁 Structure Preservation**: Maintains original directory hierarchy
- **⚡ Batch Processing**: Efficiently processes entire datasets
- **🎛️ Interactive Configuration**: User-friendly parameter selection
- **📊 Progress Tracking**: Real-time processing feedback
- **🔒 Safe Operations**: Non-destructive augmentation with separate output
- **🎯 Flexible Output**: Configurable number of variants per image

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Virtual environment support (recommended)

### Installation

1. **Clone or download** the `main.py` script

2. **Create a virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv augmentation-cli

   # Activate virtual environment
   # On Windows:
   augmentation-cli\Scripts\activate

   # On macOS/Linux:
   source augmentation-cli/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install Pillow numpy
   ```

4. **Verify installation**:
   ```bash
   python -c "import PIL, numpy; print('Dependencies installed successfully!')"
   ```

### Basic Usage

```bash
# Make sure your virtual environment is activated
source augmentation-cli/bin/activate  # macOS/Linux
# OR
augmentation-cli\Scripts\activate     # Windows

# Navigate to your script directory
cd /path/to/script

# Run the augmentation tool
python main.py /path/to/your/dataset
```

## 📖 Detailed Usage

### Command Line Arguments

```bash
python main.py [dataset_path] [options]
```

**Arguments:**
- `dataset_path` - Path to your image dataset directory (required)

**Options:**
- `--interactive, -i` - Run in interactive mode (default behavior)
- `--version, -v` - Show version information
- `--help, -h` - Display help message

### Examples

```bash
# Activate virtual environment first
source augmentation-cli/bin/activate  # macOS/Linux
# OR
augmentation-cli\Scripts\activate     # Windows

# Basic augmentation
python main.py ./my_images

# Augment dataset in current directory
python main.py .

# Get help
python main.py --help

# Deactivate virtual environment when done
deactivate
```

## 🔧 Available Transformations

| Transformation | Description | Parameters |
|----------------|-------------|------------|
| **Rotation** | Rotate image by specified degrees | -180° to 180° |
| **Horizontal Flip** | Mirror image horizontally | On/Off |
| **Vertical Flip** | Mirror image vertically | On/Off |
| **Brightness** | Adjust image brightness | 0.1 to 3.0 (1.0 = original) |
| **Contrast** | Adjust image contrast | 0.1 to 3.0 (1.0 = original) |
| **Saturation** | Adjust color saturation | 0.1 to 3.0 (1.0 = original) |
| **Blur** | Apply Gaussian blur | 0.1 to 5.0 radius |
| **Noise** | Add random noise | 0.1 to 1.0 intensity |
| **Zoom** | Zoom in/out with cropping | 0.5 to 2.0 factor |
| **Random Crop** | Crop and resize to original | 0.1 to 0.9 percentage |
| **Color Shift** | Shift individual color channels | -50 to 50 shift value |

## 📋 Interactive Workflow

The tool guides you through a simple 4-step process:

### Step 1: Dataset Validation
```
✅ Found 150 image files in the dataset
```

### Step 2: Transformation Selection
```
🔧 Available Transformations:
==================================================
 1. rotation        - Rotate image by specified degrees (-180 to 180)
 2. flip_horizontal - Flip image horizontally
 3. brightness      - Adjust brightness (0.1 to 3.0, 1.0 = original)
 ...

📋 Select transformations (enter numbers separated by commas, e.g., 1,3,5):
Your selection: 1,2,4,6
```

### Step 3: Parameter Configuration
```
rotation - degrees (-180 to 180):
Enter degrees (default: 30): 45

brightness - factor (0.1 to 3.0):
Enter factor (default: 1.5): 1.8
```

### Step 4: Sample Configuration
```
🔢 How many augmented versions per original image? (1-10)
Number of samples: 3
```

## 📁 Output Structure

The tool creates a timestamped directory alongside your original dataset:

```
your_dataset/
  ├── class1/
  └── class2/
augmented_20240619_143022/
  ├── class1/
    ├── original_image1.jpg
    ├── aug_1_rotation_brightness_image1.jpg
    ├── aug_2_flip_horizontal_contrast_image1.jpg
    └── aug_3_noise_blur_image1.jpg
  ├── class2/
    ├── original_image2.jpg
    ├── aug_1_rotation_brightness_image2.jpg
    ├── aug_2_flip_horizontal_contrast_image2.jpg
    └── aug_3_noise_blur_image2.jpg
```

**Naming Convention:**
- `original_` - Copy of original image
- `aug_[number]_[transforms]_[filename]` - Augmented versions

## 🎯 Use Cases

### Machine Learning
- **Training Data Expansion**: Increase dataset size for better model performance
- **Data Imbalance**: Balance classes by augmenting underrepresented samples
- **Overfitting Prevention**: Add variation to prevent memorization

### Computer Vision
- **Robustness Testing**: Test model performance under various conditions
- **Domain Adaptation**: Simulate different imaging conditions
- **Preprocessing Pipeline**: Standardize augmentation across projects

### Research & Development
- **Ablation Studies**: Test impact of different augmentation strategies
- **Baseline Creation**: Generate consistent augmented datasets
- **Rapid Prototyping**: Quick dataset variations for experiments

## ⚙️ Advanced Configuration

### Batch Processing Tips

For large datasets:
1. **Start Small**: Test with a subset first
2. **Monitor Resources**: Check available disk space
3. **Use Moderate Settings**: Balance quality vs. processing time

### Custom Workflows

You can modify the script for specific needs:
- Add new transformation functions
- Adjust parameter ranges
- Implement custom naming schemes
- Add format-specific optimizations

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'PIL' or 'numpy'"**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install Pillow numpy`
- Verify installation: `python -c "import PIL, numpy"`

**"Virtual environment not found"**
- Recreate virtual environment: `python -m venv augmentation-cli`
- Check Python version compatibility
- Ensure proper activation command for your OS

**"No supported image files found"**
- Check file extensions are supported
- Verify images aren't corrupted
- Ensure proper directory path

**"Permission denied"**
- Check write permissions in target directory
- Run with appropriate user privileges
- Verify disk space availability

**"Out of memory errors"**
- Process smaller batches
- Reduce number of samples per image
- Close other memory-intensive applications

### Virtual Environment Tips

**Managing Dependencies:**
```bash
# Create requirements file
pip freeze > requirements.txt

# Install from requirements file
pip install -r requirements.txt

# Update packages
pip install --upgrade Pillow numpy
```

**Environment Management:**
```bash
# Check active environment
which python  # macOS/Linux
where python   # Windows

# List installed packages
pip list

# Remove virtual environment
rm -rf augmentation-cli  # macOS/Linux
rmdir /s augmentation-cli  # Windows
```

### Performance Tips

- **SSD Storage**: Use SSD for faster I/O operations
- **Sufficient RAM**: Ensure adequate memory for image processing
- **Reasonable Batch Sizes**: Don't overwhelm system resources

## 📊 Technical Specifications

### Dependencies
- **Pillow (PIL)**: Image processing and manipulation
- **NumPy**: Numerical operations and array handling
- **Python Standard Library**: File operations, argument parsing

### Supported Formats
- **Input**: JPG, JPEG, PNG, BMP, TIFF, WEBP
- **Output**: Same format as input (with quality preservation)

### System Requirements
- **Python**: 3.7 or higher
- **RAM**: Minimum 4GB (8GB+ recommended for large datasets)
- **Storage**: 2-3x original dataset size for output

## 🤝 Contributing

Contributions are welcome! Here are some ways to help:

- **Bug Reports**: Submit detailed issue reports
- **Feature Requests**: Suggest new transformations or features
- **Code Improvements**: Optimize performance or add functionality
- **Documentation**: Improve or translate documentation

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## 📞 Support

If you encounter issues or have questions:

1. **Check Documentation**: Review this README and inline help
2. **Search Issues**: Look for similar problems in existing issues
3. **Create Issue**: Submit detailed bug report or feature request
4. **Community**: Join discussions and share experiences

## 🚀 Roadmap

Future enhancements planned:

- **Video Augmentation**: Support for video file processing
- **Advanced Transforms**: Perspective correction, elastic deformation
- **Batch Configuration**: Save/load augmentation configurations
- **Integration APIs**: REST API for programmatic access
- **GUI Version**: Graphical interface for non-technical users
- **Cloud Integration**: Support for cloud storage services

---

## 📈 Version History

### v1.0.0 (Current)
- Initial release with 11 transformation types
- Interactive CLI interface
- Batch processing capabilities
- Comprehensive error handling
- Progress tracking and validation

---

**Made with ❤️ for the ML/AI community**

*Happy augmenting! 🎨📸*
