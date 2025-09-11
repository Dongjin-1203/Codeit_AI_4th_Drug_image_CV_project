import os
import glob
import json
from PIL import Image

def validate_images(data_path, min_size=100):
    """Validate images in dataset"""
    print("Validating images...")
    
    train_images = glob.glob(os.path.join(data_path, "train_images", "*.png"))
    test_images = glob.glob(os.path.join(data_path, "test_images", "*.png"))
    all_images = train_images + test_images
    
    valid_count = 0
    issues = []
    
    for img_path in all_images:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Check minimum size
                if width < min_size or height < min_size:
                    issues.append(f"Small image: {img_path} ({width}x{height})")
                    continue
                    
                # Check format
                if img.mode not in ['RGB', 'L']:
                    issues.append(f"Invalid format: {img_path} ({img.mode})")
                    continue
                    
                valid_count += 1
                
        except Exception as e:
            issues.append(f"Corrupted image: {img_path} - {str(e)}")
    
    print(f"Images: {valid_count}/{len(all_images)} valid")
    if issues:
        print(f"Found {len(issues)} issues")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
    
    return valid_count, len(all_images), issues

def validate_annotations(data_path):
    """Validate annotation files"""
    print("Validating annotations...")
    
    train_images = glob.glob(os.path.join(data_path, "train_images", "*.png"))
    valid_count = 0
    issues = []
    
    for img_path in train_images:
        img_name = os.path.basename(img_path).replace('.png', '')
        json_files = glob.glob(
            os.path.join(data_path, "train_annotations", "*", "*", f"{img_name}.json")
        )
        
        if not json_files:
            issues.append(f"Missing annotation: {img_name}")
            continue
            
        try:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                json.load(f)
            valid_count += 1
        except Exception as e:
            issues.append(f"Invalid JSON: {json_files[0]} - {str(e)}")
    
    print(f"Annotations: {valid_count}/{len(train_images)} valid")
    if issues:
        print(f"Found {len(issues)} issues")
        for issue in issues[:5]:
            print(f"  - {issue}")
    
    return valid_count, len(train_images), issues

def validate_dataset(data_path):
    """Run complete dataset validation"""
    print("=" * 50)
    print("Dataset Validation")
    print("=" * 50)
    
    img_valid, img_total, img_issues = validate_images(data_path)
    ann_valid, ann_total, ann_issues = validate_annotations(data_path)
    
    print(f"\nValidation Summary:")
    print(f"Images: {img_valid}/{img_total} valid ({img_valid/img_total*100:.1f}%)")
    print(f"Annotations: {ann_valid}/{ann_total} valid ({ann_valid/ann_total*100:.1f}%)")
    
    return {
        'images': {'valid': img_valid, 'total': img_total, 'issues': img_issues},
        'annotations': {'valid': ann_valid, 'total': ann_total, 'issues': ann_issues}
    }