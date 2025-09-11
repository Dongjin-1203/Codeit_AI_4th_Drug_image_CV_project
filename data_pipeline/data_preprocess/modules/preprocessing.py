import os
import json
import glob
from PIL import Image

# =============================================================================
# Image Preprocessing Functions
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
# Pipeline wrapper functions
# =============================================================================

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