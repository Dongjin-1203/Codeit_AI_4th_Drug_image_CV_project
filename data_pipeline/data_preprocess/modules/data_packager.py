import os
import json
import shutil
import zipfile
import hashlib
from datetime import datetime
import glob

def package_for_delivery(dataset_path, analysis_report_path, output_path, package_name="pill_dataset"):
    """
    Package final dataset for modeling team delivery
    
    Args:
        dataset_path: Path to final YOLO dataset
        analysis_report_path: Path to analysis reports
        output_path: Output directory for packaged dataset
        package_name: Name for the package
    """
    print(f"Packaging dataset for delivery: {package_name}")
    
    # Create package directory
    package_dir = os.path.join(output_path, package_name)
    os.makedirs(package_dir, exist_ok=True)
    
    # Copy dataset
    dataset_dest = os.path.join(package_dir, "dataset")
    if os.path.exists(dataset_dest):
        shutil.rmtree(dataset_dest)
    shutil.copytree(dataset_path, dataset_dest)
    
    # Copy reports
    reports_dest = os.path.join(package_dir, "reports")
    if os.path.exists(analysis_report_path):
        if os.path.exists(reports_dest):
            shutil.rmtree(reports_dest)
        shutil.copytree(analysis_report_path, reports_dest)
    
    # Generate delivery documents
    _create_readme(package_dir, dataset_path)
    _create_usage_guide(package_dir)
    _create_validation_script(package_dir)
    _create_metadata(package_dir, dataset_path, analysis_report_path)
    
    # Validate package
    validation_result = _validate_package(package_dir)
    
    # Create compressed archive
    archive_path = _create_archive(package_dir, output_path, package_name)
    
    print(f"Package created successfully: {archive_path}")
    
    return {
        'package_path': package_dir,
        'archive_path': archive_path,
        'validation_passed': validation_result,
        'package_size_mb': _get_directory_size(package_dir) / (1024*1024)
    }

def _create_readme(package_dir, dataset_path):
    """Create README.md for the dataset package"""
    readme_content = f"""# Pill Detection Dataset

## Dataset Information
- **Creation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Format**: YOLO (You Only Look Once)
- **Image Size**: 1280x1280 pixels
- **Purpose**: Pill detection and classification

## Directory Structure
```
dataset/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
├── labels/
│   ├── train/          # Training labels (YOLO format)
│   ├── val/            # Validation labels
│   └── test/           # Test labels
└── dataset.yaml        # Dataset configuration

reports/
├── dataset_analysis_report.json    # Detailed analysis
└── ...

docs/
├── README.md           # This file
├── USAGE_GUIDE.md      # Usage instructions
└── validate_dataset.py # Validation script
```

## Quick Start
1. **Validation**: Run `python docs/validate_dataset.py` to verify dataset integrity
2. **Training**: Use `dataset/dataset.yaml` as config for YOLO training
3. **Analysis**: Check `reports/` folder for dataset statistics

## Dataset Statistics
- See `reports/dataset_analysis_report.json` for detailed statistics
- Run validation script for real-time verification

## Contact
For questions about this dataset, contact the data engineering team.

## Version
Dataset Version: 1.0
Processing Pipeline Version: 1.0
"""
    
    docs_dir = os.path.join(package_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    with open(os.path.join(docs_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write(readme_content)

def _create_usage_guide(package_dir):
    """Create detailed usage guide"""
    usage_content = """# Dataset Usage Guide

## For YOLO Training

### 1. Basic Training Setup
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt

# Train
results = model.train(
    data='dataset/dataset.yaml',
    epochs=100,
    imgsz=1280,
    batch=16
)
```

### 2. Dataset Loading
```python
import yaml

# Load dataset config
with open('dataset/dataset.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"Classes: {config['names']}")
print(f"Number of classes: {config['nc']}")
```

### 3. Custom Data Loader
```python
import os
from PIL import Image

def load_dataset_sample(split='train', index=0):
    images_dir = f'dataset/images/{split}'
    labels_dir = f'dataset/labels/{split}'
    
    image_files = sorted(os.listdir(images_dir))
    image_path = os.path.join(images_dir, image_files[index])
    label_path = os.path.join(labels_dir, image_files[index].replace('.png', '.txt'))
    
    image = Image.open(image_path)
    
    with open(label_path, 'r') as f:
        labels = f.read().strip().split('\\n')
    
    return image, labels

# Example usage
image, labels = load_dataset_sample('train', 0)
print(f"Image size: {image.size}")
print(f"Number of objects: {len(labels)}")
```

## Data Validation

### Quick Validation
```bash
python docs/validate_dataset.py
```

### Manual Validation
```python
import os
import glob

def validate_split(split_name):
    images = glob.glob(f'dataset/images/{split_name}/*.png')
    labels = glob.glob(f'dataset/labels/{split_name}/*.txt')
    
    print(f"{split_name}: {len(images)} images, {len(labels)} labels")
    
    # Check for missing pairs
    image_names = {os.path.basename(f).replace('.png', '') for f in images}
    label_names = {os.path.basename(f).replace('.txt', '') for f in labels}
    
    missing_labels = image_names - label_names
    missing_images = label_names - image_names
    
    if missing_labels:
        print(f"Missing labels: {len(missing_labels)}")
    if missing_images:
        print(f"Missing images: {len(missing_images)}")
    
    return len(missing_labels) == 0 and len(missing_images) == 0

# Validate all splits
for split in ['train', 'val', 'test']:
    validate_split(split)
```

## Troubleshooting

### Common Issues
1. **Path Problems**: Use absolute paths in dataset.yaml if needed
2. **Image Format**: All images should be PNG format, 1280x1280
3. **Label Format**: YOLO format (class_id center_x center_y width height)

### Performance Tips
1. **Batch Size**: Adjust based on GPU memory
2. **Image Size**: 1280 is optimized for this dataset
3. **Epochs**: Start with 100, adjust based on validation loss

## File Formats

### YOLO Label Format
```
# Each line: class_id center_x center_y width height
0 0.5 0.5 0.3 0.4
1 0.2 0.8 0.1 0.2
```

### dataset.yaml Format
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 15  # number of classes
names: ['class1', 'class2', ...]  # class names
```
"""
    
    docs_dir = os.path.join(package_dir, "docs")
    with open(os.path.join(docs_dir, "USAGE_GUIDE.md"), 'w', encoding='utf-8') as f:
        f.write(usage_content)

def _create_validation_script(package_dir):
    """Create validation script for modeling team"""
    script_content = '''#!/usr/bin/env python3
"""
Dataset Validation Script for Modeling Team

This script validates the dataset package integrity.
Run this before starting any training.

Usage: python validate_dataset.py
"""

import os
import glob
import yaml
from collections import Counter

def validate_dataset():
    """Validate complete dataset package"""
    print("Validating dataset package...")
    
    # Check directory structure
    required_dirs = [
        "dataset/images/train",
        "dataset/images/val", 
        "dataset/images/test",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"Missing directories: {missing_dirs}")
        return False
    
    # Check dataset.yaml
    yaml_path = "dataset/dataset.yaml"
    if not os.path.exists(yaml_path):
        print("Missing dataset.yaml")
        return False
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Dataset config loaded: {config['nc']} classes")
    
    # Validate each split
    splits = ["train", "val", "test"]
    total_images = 0
    total_labels = 0
    
    for split in splits:
        images = glob.glob(f"dataset/images/{split}/*.png")
        labels = glob.glob(f"dataset/labels/{split}/*.txt")
        
        print(f"{split}: {len(images)} images, {len(labels)} labels")
        
        if len(images) != len(labels):
            print(f"Mismatch in {split}: {len(images)} images vs {len(labels)} labels")
            return False
        
        total_images += len(images)
        total_labels += len(labels)
    
    print(f"Total: {total_images} images, {total_labels} labels")
    
    # Sample label validation
    sample_label = glob.glob("dataset/labels/train/*.txt")[0]
    with open(sample_label, 'r') as f:
        lines = f.readlines()
    
    print(f"Sample label has {len(lines)} objects")
    
    print("Dataset validation PASSED")
    return True

def analyze_class_distribution():
    """Analyze class distribution across splits"""
    print("\\nAnalyzing class distribution...")
    
    all_classes = []
    splits = ["train", "val", "test"]
    
    for split in splits:
        labels = glob.glob(f"dataset/labels/{split}/*.txt")
        split_classes = []
        
        for label_file in labels:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        split_classes.append(class_id)
        
        class_counts = Counter(split_classes)
        print(f"{split}: {len(split_classes)} objects, {len(class_counts)} unique classes")
        all_classes.extend(split_classes)
    
    overall_counts = Counter(all_classes)
    print(f"Overall: {len(all_classes)} objects, {len(overall_counts)} unique classes")
    print(f"Top 5 classes: {dict(overall_counts.most_common(5))}")

if __name__ == "__main__":
    success = validate_dataset()
    if success:
        analyze_class_distribution()
        print("\\nDataset is ready for training!")
    else:
        print("\\nDataset validation failed. Please check the issues above.")
'''
    
    docs_dir = os.path.join(package_dir, "docs")
    with open(os.path.join(docs_dir, "validate_dataset.py"), 'w', encoding='utf-8') as f:
        f.write(script_content)

def _create_metadata(package_dir, dataset_path, analysis_report_path):
    """Create package metadata"""
    metadata = {
        'package_info': {
            'name': 'Pill Detection Dataset',
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'format': 'YOLO',
            'image_size': '1280x1280'
        },
        'processing_pipeline': {
            'steps': [
                'Data Quality Validation',
                'Image Preprocessing (resize, padding)',
                'Annotation Preprocessing (COCO to YOLO)',
                'Dataset Splitting (train/val/test)',
                'Quality Analysis',
                'Packaging for Delivery'
            ],
            'image_preprocessing': {
                'target_size': 1280,
                'padding_color': [0, 0, 0],
                'aspect_ratio_preserved': True
            }
        },
        'file_counts': _count_files(dataset_path),
        'checksums': _generate_checksums(package_dir)
    }
    
    docs_dir = os.path.join(package_dir, "docs")
    with open(os.path.join(docs_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def _count_files(dataset_path):
    """Count files in dataset"""
    if not os.path.exists(dataset_path):
        return {}
    
    counts = {}
    splits = ["train", "val", "test"]
    
    for split in splits:
        images = len(glob.glob(os.path.join(dataset_path, "images", split, "*.png")))
        labels = len(glob.glob(os.path.join(dataset_path, "labels", split, "*.txt")))
        counts[split] = {'images': images, 'labels': labels}
    
    return counts

def _generate_checksums(package_dir):
    """Generate checksums for important files"""
    checksums = {}
    
    important_files = [
        "dataset/dataset.yaml",
        "docs/README.md",
        "docs/validate_dataset.py"
    ]
    
    for file_path in important_files:
        full_path = os.path.join(package_dir, file_path)
        if os.path.exists(full_path):
            with open(full_path, 'rb') as f:
                content = f.read()
                checksums[file_path] = hashlib.md5(content).hexdigest()
    
    return checksums

def _validate_package(package_dir):
    """Validate the created package"""
    print("Validating package...")
    
    # Check required files/directories
    required_items = [
        "dataset",
        "dataset/dataset.yaml",
        "dataset/images/train",
        "dataset/images/val", 
        "dataset/images/test",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test",
        "docs/README.md",
        "docs/validate_dataset.py"
    ]
    
    for item in required_items:
        item_path = os.path.join(package_dir, item)
        if not os.path.exists(item_path):
            print(f"Missing required item: {item}")
            return False
    
    print("Package validation passed")
    return True

def _create_archive(package_dir, output_path, package_name):
    """Create compressed archive of the package"""
    archive_path = os.path.join(output_path, f"{package_name}.zip")
    
    print(f"Creating archive: {archive_path}")
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, package_dir)
                zipf.write(file_path, arcname)
    
    return archive_path

def _get_directory_size(directory):
    """Get total size of directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size

def create_delivery_checklist(package_path):
    """Create delivery checklist for handoff"""
    checklist = """# 데이터셋 전달 체크리스트

## 전달 전 검증
- [ ] 필수 디렉토리 모두 존재
- [ ] 모든 분할에서 이미지-라벨 쌍 일치 확인
- [ ] dataset.yaml 파일 유효성 검증
- [ ] 검증 스크립트 정상 실행 확인
- [ ] 분석 보고서 포함 확인
- [ ] README 및 사용 가이드 완성 확인

## 패키지 구성 요소
- [ ] YOLO 구조의 dataset/ 폴더
- [ ] 분석 결과가 포함된 reports/ 폴더
- [ ] 문서가 포함된 docs/ 폴더
- [ ] 모든 파일 접근 및 읽기 가능

## 모델링팀 인수인계
- [ ] 패키지 위치 공유 완료
- [ ] 접근 권한 확인 완료
- [ ] 연락처 정보 제공 완료
- [ ] 질문/이슈 소통 채널 설정 완료

## 전달 후 확인사항
- [ ] 모델링팀 수령 확인
- [ ] 모델링팀의 초기 검증 완료
- [ ] 발생한 이슈 해결 완료

## 패키지 정보
- 생성 일자: {creation_date}
- 패키지 크기: {package_size} MB
- 총 이미지 수: {total_images}
- 총 클래스 수: {total_classes}

## 연락처
데이터 엔지니어링팀: [연락처 정보]
패키지 생성자: [담당자 이름]
"""
    
    checklist_path = os.path.join(package_path, "docs", "DELIVERY_CHECKLIST.md")
    with open(checklist_path, 'w', encoding='utf-8') as f:
        f.write(checklist.format(
            creation_date=datetime.now().strftime('%Y-%m-%d'),
            package_size="[calculated]",
            total_images="[from metadata]",
            total_classes="[from dataset.yaml]"
        ))
    
    return checklist_path

# Integration function for pipeline
def run_final_packaging(dataset_path, analysis_report_path, output_path, package_name="pill_dataset_v1"):
    """Run complete packaging pipeline"""
    print("=" * 50)
    print("Final Dataset Packaging for Modeling Team")
    print("=" * 50)
    
    result = package_for_delivery(dataset_path, analysis_report_path, output_path, package_name)
    
    # Create delivery checklist
    checklist_path = create_delivery_checklist(result['package_path'])
    
    print(f"\nPackaging Summary:")
    print(f"Package Size: {result['package_size_mb']:.1f} MB")
    print(f"Validation: {'PASSED' if result['validation_passed'] else 'FAILED'}")
    print(f"Archive: {result['archive_path']}")
    print(f"Checklist: {checklist_path}")
    
    return {
        **result,
        'checklist_path': checklist_path,
        'ready_for_delivery': result['validation_passed']
    }