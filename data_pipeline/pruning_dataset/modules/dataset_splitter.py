import os
import glob
import shutil
import random
from collections import defaultdict

def create_small_dataset(source_path, output_path, train_size, test_size, sampling_strategy="balanced"):
    """Create small dataset from original data"""
    print(f"Creating dataset: {train_size} train, {test_size} test -> {output_path}")
    
    # Setup directories
    _setup_directories(output_path)
    
    # Extract and copy data
    train_pairs = _extract_train_pairs(source_path, train_size, sampling_strategy)
    _copy_train_data(train_pairs, source_path, output_path)
    
    test_files = _extract_test_files(source_path, test_size)
    _copy_test_data(test_files, output_path)
    
    # Verify result
    _verify_dataset(output_path)
    print(f"Dataset created successfully at {output_path}")

def _setup_directories(output_path):
    """Create output directory structure"""
    dirs = ["train_images", "test_images", "train_annotations"]
    for d in dirs:
        os.makedirs(os.path.join(output_path, d), exist_ok=True)

def _extract_train_pairs(source_path, train_size, strategy):
    """Extract training image-annotation pairs"""
    train_images = glob.glob(os.path.join(source_path, "train_images", "*.png"))
    annotation_folders = glob.glob(os.path.join(source_path, "train_annotations", "*"))
    
    # Find valid pairs
    valid_pairs = []
    for img_path in train_images:
        img_name = os.path.basename(img_path).replace('.png', '')
        json_path = _find_annotation(img_name, annotation_folders)
        
        if json_path:
            valid_pairs.append({
                'image': img_path,
                'annotation': json_path,
                'name': img_name
            })
    
    # Sample based on strategy
    if strategy == "balanced":
        return _balanced_sample(valid_pairs, train_size)
    elif strategy == "quality":
        return _quality_sample(valid_pairs, train_size)
    else:  # random
        return random.sample(valid_pairs, min(train_size, len(valid_pairs)))

def _find_annotation(img_name, annotation_folders):
    """Find matching annotation file"""
    for folder in annotation_folders:
        json_files = glob.glob(os.path.join(folder, "*", f"{img_name}.json"))
        if json_files:
            return json_files[0]
    return None

def _balanced_sample(pairs, target_size):
    """Sample balanced by pill codes"""
    # Group by pill code
    groups = defaultdict(list)
    for pair in pairs:
        try:
            pill_code = pair['name'].split('-')[1]  # Extract K-003544 -> 003544
            groups[pill_code].append(pair)
        except:
            groups['unknown'].append(pair)
    
    # Sample from each group
    samples_per_group = max(1, target_size // len(groups))
    selected = []
    
    for code, group_pairs in groups.items():
        sample_count = min(samples_per_group, len(group_pairs))
        selected.extend(random.sample(group_pairs, sample_count))
    
    # Adjust to target size
    if len(selected) < target_size:
        remaining = [p for p in pairs if p not in selected]
        additional_count = min(target_size - len(selected), len(remaining))
        selected.extend(random.sample(remaining, additional_count))
    elif len(selected) > target_size:
        selected = random.sample(selected, target_size)
    
    return selected

def _quality_sample(pairs, target_size):
    """Sample by file size (larger = better quality)"""
    pairs_with_size = [(pair, os.path.getsize(pair['image'])) for pair in pairs]
    pairs_with_size.sort(key=lambda x: x[1], reverse=True)
    return [pair for pair, _ in pairs_with_size[:target_size]]

def _copy_train_data(pairs, source_path, output_path):
    """Copy selected training data"""
    for pair in pairs:
        # Copy image
        img_src = pair['image']
        img_dst = os.path.join(output_path, "train_images", os.path.basename(img_src))
        shutil.copy2(img_src, img_dst)
        
        # Copy annotation (preserve directory structure)
        json_src = pair['annotation']
        rel_path = os.path.relpath(json_src, os.path.join(source_path, "train_annotations"))
        json_dst = os.path.join(output_path, "train_annotations", rel_path)
        
        os.makedirs(os.path.dirname(json_dst), exist_ok=True)
        shutil.copy2(json_src, json_dst)

def _extract_test_files(source_path, test_size):
    """Extract test image files"""
    test_images = glob.glob(os.path.join(source_path, "test_images", "*.png"))
    return random.sample(test_images, min(test_size, len(test_images)))

def _copy_test_data(files, output_path):
    """Copy selected test data"""
    for img_src in files:
        img_dst = os.path.join(output_path, "test_images", os.path.basename(img_src))
        shutil.copy2(img_src, img_dst)

def _verify_dataset(output_path):
    """Verify created dataset"""
    train_count = len(glob.glob(os.path.join(output_path, "train_images", "*.png")))
    test_count = len(glob.glob(os.path.join(output_path, "test_images", "*.png")))
    annotation_count = len(glob.glob(os.path.join(output_path, "train_annotations", "*", "*", "*.json")))
    
    print(f"Result: {train_count} train, {test_count} test, {annotation_count} annotations")