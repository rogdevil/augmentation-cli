#!/usr/bin/env python3
"""
Data Augmentation CLI Tool
A comprehensive command-line interface for augmenting image datasets
"""

import os
import sys
import argparse
import json
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import List, Dict, Tuple, Optional
import shutil
from datetime import datetime

class DataAugmenter:
    """Main class for handling data augmentation operations"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path).resolve()  # Resolve absolute path
        # Create output directory with shorter name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_path = self.dataset_path.parent / f"augmented_{timestamp}"
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.transformations = {
            'rotation': self._rotate,
            'flip_horizontal': self._flip_horizontal,
            'flip_vertical': self._flip_vertical,
            'brightness': self._adjust_brightness,
            'contrast': self._adjust_contrast,
            'saturation': self._adjust_saturation,
            'blur': self._apply_blur,
            'noise': self._add_noise,
            'zoom': self._zoom,
            'crop': self._random_crop,
            'color_shift': self._color_shift
        }

    def validate_dataset(self) -> bool:
        """Validate if the dataset path exists and contains images"""
        if not self.dataset_path.exists():
            print(f"❌ Error: Dataset path '{self.dataset_path}' does not exist!")
            return False

        # Check if this is a previously augmented dataset
        if 'augmented' in self.dataset_path.name.lower():
            print(f"⚠️  Warning: This appears to be a previously augmented dataset!")
            print(f"📁 Dataset path: {self.dataset_path}")
            confirm = input("Continue anyway? This may cause issues. (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                return False

        image_files = self._get_image_files()
        if not image_files:
            print(f"❌ Error: No supported image files found in '{self.dataset_path}'!")
            print(f"Supported formats: {', '.join(self.supported_formats)}")
            return False

        print(f"✅ Found {len(image_files)} image files in the dataset")
        return True

    def _get_image_files(self) -> List[Path]:
        """Get all image files from the dataset directory"""
        image_files = []

        # First, try to get images directly from the root directory
        for file in self.dataset_path.iterdir():
            if file.is_file() and file.suffix.lower() in self.supported_formats:
                image_files.append(file)

        # If no images in root, search subdirectories (but avoid deep nesting)
        if not image_files:
            for root, dirs, files in os.walk(self.dataset_path):
                root_path = Path(root)

                # Skip if path is too deep (more than 3 levels)
                try:
                    relative_path = root_path.relative_to(self.dataset_path)
                    if len(relative_path.parts) > 3:
                        continue
                except ValueError:
                    continue

                for file in files:
                    file_path = root_path / file
                    if file_path.suffix.lower() in self.supported_formats:
                        # Skip already processed augmented files
                        if not any(prefix in file.lower() for prefix in ['aug_', 'original_']):
                            image_files.append(file_path)

        return image_files

    def display_transformation_options(self):
        """Display available transformation options"""
        print("\n🔧 Available Transformations:")
        print("=" * 50)
        transformations_info = {
            'rotation': 'Rotate image by specified degrees (-180 to 180)',
            'flip_horizontal': 'Flip image horizontally',
            'flip_vertical': 'Flip image vertically',
            'brightness': 'Adjust brightness (0.1 to 3.0, 1.0 = original)',
            'contrast': 'Adjust contrast (0.1 to 3.0, 1.0 = original)',
            'saturation': 'Adjust color saturation (0.1 to 3.0, 1.0 = original)',
            'blur': 'Apply Gaussian blur (radius 0.1 to 5.0)',
            'noise': 'Add random noise (intensity 0.1 to 1.0)',
            'zoom': 'Zoom in/out (0.5 to 2.0, 1.0 = original)',
            'crop': 'Random crop (percentage 0.1 to 0.9)',
            'color_shift': 'Shift color channels (shift value -50 to 50)'
        }

        for i, (key, description) in enumerate(transformations_info.items(), 1):
            print(f"{i:2d}. {key:15s} - {description}")

    def get_user_selections(self) -> Dict:
        """Interactive selection of transformations and parameters"""
        self.display_transformation_options()

        print(f"\n📋 Select transformations (enter numbers separated by commas, e.g., 1,3,5):")
        print("    Or enter 'all' to select all transformations")

        user_input = input("Your selection: ").strip()

        if user_input.lower() == 'all':
            selected_transformations = list(self.transformations.keys())
        else:
            try:
                indices = [int(x.strip()) for x in user_input.split(',')]
                transformation_list = list(self.transformations.keys())
                selected_transformations = [transformation_list[i-1] for i in indices if 1 <= i <= len(transformation_list)]
            except (ValueError, IndexError):
                print("❌ Invalid selection! Using default transformations.")
                selected_transformations = ['rotation', 'flip_horizontal', 'brightness']

        print(f"\n✅ Selected transformations: {', '.join(selected_transformations)}")

        # Get parameters for each transformation
        transformation_params = {}
        for transform in selected_transformations:
            transformation_params[transform] = self._get_transformation_parameters(transform)

        # Get number of augmented samples per original image
        print(f"\n🔢 How many augmented versions per original image? (1-10)")
        try:
            num_samples = int(input("Number of samples: ").strip())
            num_samples = max(1, min(10, num_samples))
        except ValueError:
            num_samples = 2
            print("Invalid input! Using default: 2 samples per image")

        return {
            'transformations': transformation_params,
            'num_samples': num_samples
        }

    def _get_transformation_parameters(self, transform_name: str) -> Dict:
        """Get parameters for specific transformation"""
        param_ranges = {
            'rotation': {'min': -180, 'max': 180, 'default': 30, 'name': 'degrees'},
            'brightness': {'min': 0.1, 'max': 3.0, 'default': 1.5, 'name': 'factor'},
            'contrast': {'min': 0.1, 'max': 3.0, 'default': 1.5, 'name': 'factor'},
            'saturation': {'min': 0.1, 'max': 3.0, 'default': 1.5, 'name': 'factor'},
            'blur': {'min': 0.1, 'max': 5.0, 'default': 1.0, 'name': 'radius'},
            'noise': {'min': 0.1, 'max': 1.0, 'default': 0.3, 'name': 'intensity'},
            'zoom': {'min': 0.5, 'max': 2.0, 'default': 1.2, 'name': 'factor'},
            'crop': {'min': 0.1, 'max': 0.9, 'default': 0.8, 'name': 'percentage'},
            'color_shift': {'min': -50, 'max': 50, 'default': 20, 'name': 'shift'}
        }

        if transform_name in ['flip_horizontal', 'flip_vertical']:
            return {'enabled': True}

        if transform_name in param_ranges:
            range_info = param_ranges[transform_name]
            print(f"\n  {transform_name} - {range_info['name']} ({range_info['min']} to {range_info['max']}):")
            try:
                value = float(input(f"    Enter {range_info['name']} (default: {range_info['default']}): ").strip() or range_info['default'])
                value = max(range_info['min'], min(range_info['max'], value))
            except ValueError:
                value = range_info['default']
            return {'value': value}

        return {'enabled': True}

    def augment_dataset(self, config: Dict):
        """Perform data augmentation based on configuration"""
        image_files = self._get_image_files()
        total_operations = len(image_files) * config['num_samples']

        print(f"\n🚀 Starting augmentation...")
        print(f"📊 Processing {len(image_files)} images")
        print(f"🔄 Creating {config['num_samples']} variants per image")
        print(f"📈 Total operations: {total_operations}")
        print(f"💾 Output directory: {self.output_path}")

        # Create output directory structure
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Copy original dataset structure
        self._copy_directory_structure()

        processed = 0
        for image_file in image_files:
            try:
                self._process_single_image(image_file, config)
                processed += 1
                progress = (processed / len(image_files)) * 100
                print(f"📋 Progress: {processed}/{len(image_files)} ({progress:.1f}%)")
            except Exception as e:
                print(f"⚠️  Error processing {image_file.name}: {str(e)}")

        print(f"\n✅ Augmentation completed!")
        print(f"📁 Augmented dataset saved to: {self.output_path}")
        print(f"📊 Total images created: {processed * config['num_samples']}")

    def _copy_directory_structure(self):
        """Copy the directory structure from original to output"""
        # Simple approach: just ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories as needed during processing
        # This avoids the recursive directory issue

    def _process_single_image(self, image_path: Path, config: Dict):
        """Process a single image with augmentations"""
        try:
            # Load original image
            with Image.open(image_path) as img:
                img = img.convert('RGB')

                # Create a safe output directory structure
                try:
                    relative_path = image_path.relative_to(self.dataset_path)
                    if len(relative_path.parts) > 1:
                        # Create subdirectory structure, but keep it simple
                        safe_subdir = Path(*relative_path.parts[:-1])  # All parts except filename
                        output_dir = self.output_path / safe_subdir
                    else:
                        output_dir = self.output_path
                except (ValueError, OSError):
                    # If path issues, just use root output directory
                    output_dir = self.output_path

                # Ensure output directory exists
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save original image with simple naming
                safe_filename = self._make_safe_filename(image_path.name)
                original_output = output_dir / f"orig_{safe_filename}"
                img.save(original_output, quality=95)

                # Create augmented versions
                for i in range(config['num_samples']):
                    augmented_img = img.copy()
                    applied_transforms = []

                    # Apply random subset of selected transformations
                    available_transforms = list(config['transformations'].keys())
                    num_transforms = random.randint(1, min(3, len(available_transforms)))
                    selected_transforms = random.sample(available_transforms, num_transforms)

                    for transform_name in selected_transforms:
                        if transform_name in self.transformations:
                            try:
                                params = config['transformations'][transform_name]
                                augmented_img = self.transformations[transform_name](augmented_img, params)
                                applied_transforms.append(transform_name[:4])  # Shorten transform names
                            except Exception as e:
                                print(f"⚠️  Warning: Failed to apply {transform_name}: {str(e)}")
                                continue

                    # Save augmented image with very simple naming
                    transform_code = ''.join(applied_transforms)[:8]  # Max 8 chars
                    output_filename = f"aug{i+1}_{transform_code}_{safe_filename}"

                    # Ensure filename is not too long
                    if len(output_filename) > 50:
                        base_name = Path(safe_filename).stem[:10]  # Keep only first 10 chars
                        ext = Path(safe_filename).suffix
                        output_filename = f"aug{i+1}_{base_name}{ext}"

                    output_path = output_dir / output_filename
                    augmented_img.save(output_path, quality=95)

        except Exception as e:
            raise Exception(f"Failed to process {image_path.name}: {str(e)}")

    def _make_safe_filename(self, filename: str) -> str:
        """Create a safe filename by removing problematic characters"""
        # Remove or replace problematic characters
        safe_name = filename.replace(' ', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '._-')

        # Limit length
        if len(safe_name) > 30:
            name_part = Path(safe_name).stem[:20]
            ext_part = Path(safe_name).suffix
            safe_name = f"{name_part}{ext_part}"

        return safe_name

    # Transformation methods
    def _rotate(self, img: Image.Image, params: Dict) -> Image.Image:
        angle = random.uniform(-params['value'], params['value'])
        return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    def _flip_horizontal(self, img: Image.Image, params: Dict) -> Image.Image:
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    def _flip_vertical(self, img: Image.Image, params: Dict) -> Image.Image:
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def _adjust_brightness(self, img: Image.Image, params: Dict) -> Image.Image:
        factor = random.uniform(0.5, params['value'])
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def _adjust_contrast(self, img: Image.Image, params: Dict) -> Image.Image:
        factor = random.uniform(0.5, params['value'])
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def _adjust_saturation(self, img: Image.Image, params: Dict) -> Image.Image:
        factor = random.uniform(0.5, params['value'])
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)

    def _apply_blur(self, img: Image.Image, params: Dict) -> Image.Image:
        radius = random.uniform(0.1, params['value'])
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def _add_noise(self, img: Image.Image, params: Dict) -> Image.Image:
        np_img = np.array(img)
        noise = np.random.normal(0, params['value'] * 25, np_img.shape)
        noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def _zoom(self, img: Image.Image, params: Dict) -> Image.Image:
        zoom_factor = random.uniform(1.0, params['value'])
        width, height = img.size
        new_size = (int(width * zoom_factor), int(height * zoom_factor))
        zoomed = img.resize(new_size, Image.Resampling.LANCZOS)

        # Crop to original size from center
        left = (new_size[0] - width) // 2
        top = (new_size[1] - height) // 2
        return zoomed.crop((left, top, left + width, top + height))

    def _random_crop(self, img: Image.Image, params: Dict) -> Image.Image:
        width, height = img.size
        crop_factor = params['value']
        new_width = int(width * crop_factor)
        new_height = int(height * crop_factor)

        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)

        cropped = img.crop((left, top, left + new_width, top + new_height))
        return cropped.resize((width, height), Image.Resampling.LANCZOS)

    def _color_shift(self, img: Image.Image, params: Dict) -> Image.Image:
        np_img = np.array(img)
        shift_value = random.uniform(-params['value'], params['value'])

        # Random channel to shift
        channel = random.randint(0, 2)
        np_img[:, :, channel] = np.clip(np_img[:, :, channel] + shift_value, 0, 255)

        return Image.fromarray(np_img.astype(np.uint8))


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Data Augmentation CLI Tool - Augment your image datasets with various transformations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python augment_data.py /path/to/dataset
  python augment_data.py ./images --interactive

Supported image formats: JPG, JPEG, PNG, BMP, TIFF, WEBP
        """
    )

    parser.add_argument(
        'dataset_path',
        help='Path to the dataset directory containing images'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode (default)'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Data Augmentation CLI v1.0.0'
    )

    args = parser.parse_args()

    print("🎨 Data Augmentation CLI Tool")
    print("=" * 40)

    # Initialize augmenter
    augmenter = DataAugmenter(args.dataset_path)

    # Validate dataset
    if not augmenter.validate_dataset():
        sys.exit(1)

    try:
        # Get user configuration
        config = augmenter.get_user_selections()

        # Confirm before proceeding
        print(f"\n📋 Configuration Summary:")
        print(f"   Dataset: {augmenter.dataset_path}")
        print(f"   Output: {augmenter.output_path}")
        print(f"   Transformations: {len(config['transformations'])}")
        print(f"   Samples per image: {config['num_samples']}")

        confirm = input(f"\n❓ Proceed with augmentation? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            augmenter.augment_dataset(config)
        else:
            print("❌ Augmentation cancelled.")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
