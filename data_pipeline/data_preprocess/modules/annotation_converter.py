import os
import json
import glob
from PIL import Image
import numpy as np
from collections import Counter

# =============================================================================
# Data Quality Validation
# =============================================================================

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

# =============================================================================
# Image Preprocessing
# =============================================================================

def calculate_resize_params(width, height, target_size=1280):
    """Calculate resize parameters maintaining aspect ratio"""
    scale = min(target_size / width, target_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    
    return {
        'scale': scale,
        'new_size': (new_width, new_height),
        'padding': (pad_left, pad_top),
        'target_size': target_size
    }

def resize_image(image, target_size=1280):
    """Resize image with padding to maintain aspect ratio"""
    width, height = image.size
    params = calculate_resize_params(width, height, target_size)
    
    # Resize
    resized = image.resize(params['new_size'], Image.Resampling.LANCZOS)
    
    # Create new canvas with padding
    new_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    pad_left, pad_top = params['padding']
    new_image.paste(resized, (pad_left, pad_top))
    
    return new_image, params

def process_image(input_path, output_path, target_size=1280):
    """Process single image"""
    try:
        with Image.open(input_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            processed_img, params = resize_image(img, target_size)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            processed_img.save(output_path, 'PNG', quality=95)
            
            return params
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None

def update_annotation_coords(annotation_path, output_path, resize_params):
    """Update annotation coordinates for resized image"""
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        scale = resize_params['scale']
        pad_left, pad_top = resize_params['padding']
        
        # Update bbox coordinates
        if 'annotations' in data:
            for ann in data['annotations']:
                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']
                    ann['bbox'] = [
                        x * scale + pad_left,
                        y * scale + pad_top,
                        w * scale,
                        h * scale
                    ]
        
        # Update image size info
        if 'images' in data:
            for img_info in data['images']:
                img_info['width'] = resize_params['target_size']
                img_info['height'] = resize_params['target_size']
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error updating annotation {annotation_path}: {e}")
        return False

def preprocess_dataset(input_path, output_path, target_size=1280):
    """Preprocess entire dataset"""
    print(f"Preprocessing dataset: {input_path} -> {output_path}")
    
    # Create output directories
    for subdir in ['train_images', 'test_images', 'train_annotations']:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)
    
    processed_count = 0
    failed_count = 0
    
    # Process training images
    train_images = glob.glob(os.path.join(input_path, "train_images", "*.png"))
    print(f"Processing {len(train_images)} training images...")
    
    for img_path in train_images:
        img_name = os.path.basename(img_path)
        output_img_path = os.path.join(output_path, "train_images", img_name)
        
        # Process image
        params = process_image(img_path, output_img_path, target_size)
        if params is None:
            failed_count += 1
            continue
        
        # Update corresponding annotation
        img_name_no_ext = img_name.replace('.png', '')
        annotation_files = glob.glob(
            os.path.join(input_path, "train_annotations", "*", "*", f"{img_name_no_ext}.json")
        )
        
        if annotation_files:
            annotation_path = annotation_files[0]
            rel_path = os.path.relpath(annotation_path, 
                                     os.path.join(input_path, "train_annotations"))
            output_annotation_path = os.path.join(output_path, "train_annotations", rel_path)
            
            if update_annotation_coords(annotation_path, output_annotation_path, params):
                processed_count += 1
        
    # Process test images
    test_images = glob.glob(os.path.join(input_path, "test_images", "*.png"))
    print(f"Processing {len(test_images)} test images...")
    
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        output_img_path = os.path.join(output_path, "test_images", img_name)
        
        if process_image(img_path, output_img_path, target_size):
            processed_count += 1
        else:
            failed_count += 1
    
    print(f"Preprocessing complete: {processed_count} processed, {failed_count} failed")
    return processed_count, failed_count

# =============================================================================
# COCO to YOLO Conversion
# =============================================================================

def extract_pill_code(filename):
    """Extract pill code from filename"""
    name = filename.replace('.png', '').replace('.json', '')
    return name.split('_')[0] if '_' in name else name

def convert_coco_to_yolo(data_path, output_path):
    """Convert COCO format annotations to YOLO format"""
    print(f"Converting COCO to YOLO: {data_path} -> {output_path}")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Find all annotation files
    json_files = glob.glob(os.path.join(data_path, "train_annotations", "*", "*", "*.json"))
    print(f"Found {len(json_files)} annotation files")
    
    if not json_files:
        print("No annotation files found!")
        return
    
    # Collect all pill codes for class mapping
    all_pill_codes = set()
    valid_files = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'images' in data and 'annotations' in data:
                for img_info in data['images']:
                    pill_code = extract_pill_code(img_info['file_name'])
                    all_pill_codes.add(pill_code)
                valid_files.append((json_file, data))
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    # Create class mapping
    sorted_codes = sorted(list(all_pill_codes))
    code_to_id = {code: i for i, code in enumerate(sorted_codes)}
    
    print(f"Found {len(sorted_codes)} unique pill codes")
    
    # Save class mapping
    classes_file = os.path.join(output_path, 'classes.txt')
    with open(classes_file, 'w', encoding='utf-8') as f:
        for code in sorted_codes:
            f.write(f"{code}\n")
    
    # Convert annotations to YOLO format
    converted_count = 0
    
    for json_file, coco_data in valid_files:
        images_by_id = {img['id']: img for img in coco_data['images']}
        
        for img_info in coco_data['images']:
            img_id = img_info['id']
            filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Find annotations for this image
            image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
            
            if image_annotations:
                # Create YOLO label file
                base_name = filename.replace('.png', '').replace('.jpg', '')
                yolo_file = os.path.join(output_path, f"{base_name}.txt")
                
                pill_code = extract_pill_code(filename)
                class_id = code_to_id.get(pill_code, 0)
                
                with open(yolo_file, 'w') as f:
                    for ann in image_annotations:
                        # Convert COCO bbox to YOLO format
                        x, y, w, h = ann['bbox']
                        
                        # YOLO format: center_x, center_y, width, height (normalized)
                        center_x = (x + w/2) / img_width
                        center_y = (y + h/2) / img_height
                        width_rel = w / img_width
                        height_rel = h / img_height
                        
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width_rel:.6f} {height_rel:.6f}\n")
                
                converted_count += 1
    
    print(f"Conversion complete: {converted_count} images converted")
    print(f"Classes file: {classes_file}")
    
    return {
        'converted_count': converted_count,
        'total_classes': len(sorted_codes),
        'classes_file': classes_file,
        'class_mapping': code_to_id
    }

def run_conversion(data_path, output_path):
    """Run COCO to YOLO conversion"""
    print("=" * 50)
    print("COCO to YOLO Conversion")
    print("=" * 50)
    
    result = convert_coco_to_yolo(data_path, output_path)
    
    print(f"\nConversion Summary:")
    print(f"Converted: {result['converted_count']} images")
    print(f"Classes: {result['total_classes']}")
    
    return result

# =============================================================================
# Main Pipeline Functions
# =============================================================================

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

def run_preprocessing(input_path, output_path, target_size=1280):
    """Run complete preprocessing pipeline"""
    print("=" * 50)
    print("Dataset Preprocessing")
    print("=" * 50)
    
    processed, failed = preprocess_dataset(input_path, output_path, target_size)
    
    print(f"\nPreprocessing Summary:")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {processed/(processed+failed)*100:.1f}%")
    
    return processed, failed

def run_conversion(data_path, output_path):
    """Run COCO to YOLO conversion"""
    print("=" * 50)
    print("COCO to YOLO Conversion")
    print("=" * 50)
    
    result = convert_coco_to_yolo(data_path, output_path)
    
    print(f"\nConversion Summary:")
    print(f"Converted: {result['converted_count']} images")
    print(f"Classes: {result['total_classes']}")
    
    return result

# =============================================================================
# Quick Functions for Development
# =============================================================================

def quick_validate_prototype():
    """Quick validation for prototype data"""
    return validate_dataset("./data/prototype_data")

def quick_preprocess_prototype():
    """Quick preprocessing for prototype data"""
    return run_preprocessing(
        "./data/prototype_data", 
        "./data/prototype_data/preprocessed"
    )

def quick_convert_prototype():
    """Quick conversion for prototype data"""
    return run_conversion(
        "./data/prototype_data/preprocessed",
        "./data/prototype_data/yolo_labels"
    )

def run_full_pipeline(data_path, target_size=1280):
    """Run complete pipeline: validate -> preprocess -> convert"""
    print("Running full YOLO data pipeline...")
    
    # Step 1: Validation
    validation_result = validate_dataset(data_path)
    
    # Step 2: Preprocessing
    preprocessed_path = os.path.join(data_path, "preprocessed")
    processed, failed = run_preprocessing(data_path, preprocessed_path, target_size)
    
    # Step 3: YOLO Conversion
    yolo_path = os.path.join(data_path, "yolo_labels")
    conversion_result = run_conversion(preprocessed_path, yolo_path)
    
    print("\n" + "=" * 50)
    print("Pipeline Complete!")
    print("=" * 50)
    print(f"Preprocessed images: {preprocessed_path}")
    print(f"YOLO labels: {yolo_path}")
    print(f"Classes file: {conversion_result['classes_file']}")
    
    return {
        'validation': validation_result,
        'preprocessing': {'processed': processed, 'failed': failed},
        'conversion': conversion_result
    }

if __name__ == "__main__":
    print("YOLO Data Pipeline")
    print("Available functions:")
    print("  validate_dataset(path)           - Validate dataset")
    print("  run_preprocessing(input, output) - Preprocess images")
    print("  run_conversion(input, output)    - Convert to YOLO")
    print("  run_full_pipeline(path)          - Run complete pipeline")
    print("  quick_validate_prototype()       - Quick prototype validation")
    print("  quick_preprocess_prototype()     - Quick prototype preprocessing")
    print("  quick_convert_prototype()        - Quick prototype conversion")
    
    # Example usage:
    # result = run_full_pipeline("./data/prototype_data")