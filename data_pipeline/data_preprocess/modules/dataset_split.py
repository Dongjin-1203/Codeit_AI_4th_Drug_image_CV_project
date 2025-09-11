import os
import glob
import shutil
import random
import yaml
from collections import defaultdict

def split_dataset_for_yolo(input_path, output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split YOLO dataset into train/val/test with standard YOLO structure
    
    Args:
        input_path: Path with preprocessed images and YOLO labels
        output_path: Output path for YOLO dataset structure
        train_ratio: Training set ratio (default 0.8)
        val_ratio: Validation set ratio (default 0.1) 
        test_ratio: Test set ratio (default 0.1)
    """
    print(f"Splitting dataset: {input_path} -> {output_path}")
    print(f"Ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # Create YOLO directory structure
    _create_yolo_structure(output_path)
    
    # Get all image-label pairs
    pairs = _get_image_label_pairs(input_path)
    print(f"Found {len(pairs)} image-label pairs")
    
    # Split by pill codes for balanced distribution
    train_pairs, val_pairs, test_pairs = _stratified_split(pairs, train_ratio, val_ratio, test_ratio)
    
    # Copy files to respective directories
    _copy_split_data(train_pairs, output_path, "train")
    _copy_split_data(val_pairs, output_path, "val") 
    _copy_split_data(test_pairs, output_path, "test")
    
    # Create dataset.yaml
    _create_dataset_yaml(input_path, output_path)
    
    # Summary
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Val: {len(val_pairs)} pairs") 
    print(f"  Test: {len(test_pairs)} pairs")
    
    return {
        'train_count': len(train_pairs),
        'val_count': len(val_pairs),
        'test_count': len(test_pairs),
        'output_path': output_path
    }

def _create_yolo_structure(output_path):
    """Create standard YOLO directory structure"""
    directories = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    for dir_path in directories:
        os.makedirs(os.path.join(output_path, dir_path), exist_ok=True)

def _get_image_label_pairs(input_path):
    """Get all valid image-label pairs"""
    # Look for preprocessed images
    image_files = glob.glob(os.path.join(input_path, "train_images", "*.png"))
    
    pairs = []
    for img_path in image_files:
        img_name = os.path.basename(img_path).replace('.png', '')
        
        # Find corresponding YOLO label file
        label_path = os.path.join(input_path, "yolo_labels", f"{img_name}.txt")
        
        if os.path.exists(label_path):
            pairs.append({
                'image': img_path,
                'label': label_path,
                'name': img_name,
                'pill_code': _extract_pill_code(img_name)
            })
    
    return pairs

def _extract_pill_code(filename):
    """Extract pill code from filename for grouping"""
    try:
        # Extract from K-003544-010221-016551-027926_... format
        parts = filename.split('-')
        if len(parts) >= 2:
            return parts[1]  # Return 003544
        return filename.split('_')[0]  # Fallback
    except:
        return 'unknown'

def _stratified_split(pairs, train_ratio, val_ratio, test_ratio):
    """Split pairs while maintaining class balance"""
    # Group by pill codes
    pill_groups = defaultdict(list)
    for pair in pairs:
        pill_groups[pair['pill_code']].append(pair)
    
    train_pairs, val_pairs, test_pairs = [], [], []
    
    for pill_code, group_pairs in pill_groups.items():
        random.shuffle(group_pairs)
        
        n = len(group_pairs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_pairs.extend(group_pairs[:train_end])
        val_pairs.extend(group_pairs[train_end:val_end])
        test_pairs.extend(group_pairs[val_end:])
    
    # Shuffle final splits
    random.shuffle(train_pairs)
    random.shuffle(val_pairs) 
    random.shuffle(test_pairs)
    
    return train_pairs, val_pairs, test_pairs

def _copy_split_data(pairs, output_path, split_name):
    """Copy image-label pairs to split directory"""
    for pair in pairs:
        # Copy image
        img_src = pair['image']
        img_dst = os.path.join(output_path, "images", split_name, f"{pair['name']}.png")
        shutil.copy2(img_src, img_dst)
        
        # Copy label
        label_src = pair['label']
        label_dst = os.path.join(output_path, "labels", split_name, f"{pair['name']}.txt")
        shutil.copy2(label_src, label_dst)

def _create_dataset_yaml(input_path, output_path):
    """Create dataset.yaml file for YOLO training"""
    # Read classes from existing classes.txt
    classes_file = os.path.join(input_path, "yolo_labels", "classes.txt")
    
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        class_names = ["pill"]  # Fallback
    
    # Create dataset config
    dataset_config = {
        'path': os.path.abspath(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    # Save dataset.yaml
    yaml_path = os.path.join(output_path, "dataset.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Created dataset.yaml with {len(class_names)} classes")

def create_yolo_dataset_structure(preprocessed_path, yolo_labels_path, final_output_path):
    """
    Create final YOLO dataset from preprocessed images and labels
    
    Args:
        preprocessed_path: Path to preprocessed images
        yolo_labels_path: Path to YOLO labels
        final_output_path: Final YOLO dataset output path
    """
    print("Creating final YOLO dataset structure...")
    
    # Combine preprocessed images and YOLO labels into single input
    temp_combined_path = os.path.join(os.path.dirname(final_output_path), "temp_combined")
    _combine_images_and_labels(preprocessed_path, yolo_labels_path, temp_combined_path)
    
    # Split into train/val/test
    result = split_dataset_for_yolo(temp_combined_path, final_output_path)
    
    # Cleanup temp directory
    if os.path.exists(temp_combined_path):
        shutil.rmtree(temp_combined_path)
    
    return result

def _combine_images_and_labels(preprocessed_path, yolo_labels_path, output_path):
    """Temporarily combine images and labels for splitting"""
    os.makedirs(output_path, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(output_path, "train_images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "yolo_labels"), exist_ok=True)
    
    # Copy preprocessed images
    train_images = glob.glob(os.path.join(preprocessed_path, "train_images", "*.png"))
    for img_path in train_images:
        shutil.copy2(img_path, os.path.join(output_path, "train_images"))
    
    # Copy YOLO labels
    label_files = glob.glob(os.path.join(yolo_labels_path, "*.txt"))
    for label_path in label_files:
        if not os.path.basename(label_path) == "classes.txt":
            shutil.copy2(label_path, os.path.join(output_path, "yolo_labels"))
    
    # Copy classes.txt
    classes_src = os.path.join(yolo_labels_path, "classes.txt")
    if os.path.exists(classes_src):
        shutil.copy2(classes_src, os.path.join(output_path, "yolo_labels"))

def verify_yolo_dataset(dataset_path):
    """Verify YOLO dataset structure and content"""
    print(f"Verifying YOLO dataset: {dataset_path}")
    
    # Check directory structure
    required_dirs = ["images/train", "images/val", "images/test", 
                    "labels/train", "labels/val", "labels/test"]
    
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            print(f"Missing directory: {dir_path}")
            return False
    
    # Count files
    splits = ["train", "val", "test"]
    for split in splits:
        img_count = len(glob.glob(os.path.join(dataset_path, "images", split, "*.png")))
        label_count = len(glob.glob(os.path.join(dataset_path, "labels", split, "*.txt")))
        print(f"{split}: {img_count} images, {label_count} labels")
        
        if img_count != label_count:
            print(f"Mismatch in {split}: {img_count} images vs {label_count} labels")
    
    # Check dataset.yaml
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    if os.path.exists(yaml_path):
        print("dataset.yaml found")
    else:
        print("dataset.yaml missing")
    
    return True

# Quick function for pipeline integration
def run_dataset_split(preprocessed_path, yolo_labels_path, output_path):
    """Run complete dataset splitting pipeline"""
    print("=" * 50)
    print("Dataset Splitting and YOLO Structure Creation")
    print("=" * 50)
    
    result = create_yolo_dataset_structure(preprocessed_path, yolo_labels_path, output_path)
    verify_yolo_dataset(output_path)
    
    print(f"\nDataset Split Summary:")
    print(f"Train: {result['train_count']}")
    print(f"Val: {result['val_count']}")
    print(f"Test: {result['test_count']}")
    print(f"Output: {result['output_path']}")
    
    return result