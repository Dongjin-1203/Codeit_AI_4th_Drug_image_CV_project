import os
import glob
import json
import yaml
import numpy as np
from collections import defaultdict, Counter
from PIL import Image

def analyze_yolo_dataset(dataset_path, output_path=None):
    """
    Analyze YOLO dataset and generate quality report
    
    Args:
        dataset_path: Path to YOLO dataset (with images/, labels/, dataset.yaml)
        output_path: Path to save analysis report (optional)
    """
    print(f"Analyzing YOLO dataset: {dataset_path}")
    
    # Load dataset info
    dataset_info = _load_dataset_info(dataset_path)
    
    # Analyze each split
    analysis_results = {}
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"Analyzing {split} set...")
        split_analysis = _analyze_split(dataset_path, split, dataset_info['class_names'])
        analysis_results[split] = split_analysis
    
    # Generate overall statistics
    overall_stats = _generate_overall_stats(analysis_results, dataset_info)
    
    # Create final report
    report = {
        'dataset_info': dataset_info,
        'split_analysis': analysis_results,
        'overall_stats': overall_stats,
        'quality_metrics': _calculate_quality_metrics(analysis_results)
    }
    
    # Save report
    if output_path:
        report_path = _save_analysis_report(report, output_path)
        print(f"Analysis report saved: {report_path}")
    
    # Print summary
    _print_analysis_summary(report)
    
    return report

def _load_dataset_info(dataset_path):
    """Load basic dataset information"""
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        return {
            'total_classes': yaml_data.get('nc', 0),
            'class_names': yaml_data.get('names', []),
            'dataset_path': dataset_path
        }
    else:
        # Fallback: try to read from classes.txt
        classes_file = os.path.join(dataset_path, "classes.txt")
        if os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            class_names = []
        
        return {
            'total_classes': len(class_names),
            'class_names': class_names,
            'dataset_path': dataset_path
        }

def _analyze_split(dataset_path, split, class_names):
    """Analyze single split (train/val/test)"""
    images_path = os.path.join(dataset_path, "images", split)
    labels_path = os.path.join(dataset_path, "labels", split)
    
    # Get all files
    image_files = glob.glob(os.path.join(images_path, "*.png"))
    label_files = glob.glob(os.path.join(labels_path, "*.txt"))
    
    # Basic counts
    analysis = {
        'image_count': len(image_files),
        'label_count': len(label_files),
        'class_distribution': defaultdict(int),
        'bbox_stats': {
            'total_objects': 0,
            'avg_objects_per_image': 0,
            'bbox_sizes': [],
            'bbox_centers': []
        },
        'image_stats': {
            'resolutions': [],
            'file_sizes': []
        }
    }
    
    # Analyze images
    for img_path in image_files[:100]:  # Sample first 100 for speed
        try:
            with Image.open(img_path) as img:
                analysis['image_stats']['resolutions'].append(img.size)
                analysis['image_stats']['file_sizes'].append(os.path.getsize(img_path))
        except:
            continue
    
    # Analyze labels
    objects_per_image = []
    
    for label_path in label_files:
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            objects_count = len(lines)
            objects_per_image.append(objects_count)
            analysis['bbox_stats']['total_objects'] += objects_count
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    center_x, center_y, width, height = map(float, parts[1:5])
                    
                    # Count class distribution
                    if class_id < len(class_names):
                        class_name = class_names[class_id]
                    else:
                        class_name = f"class_{class_id}"
                    analysis['class_distribution'][class_name] += 1
                    
                    # Store bbox stats
                    analysis['bbox_stats']['bbox_sizes'].append((width, height))
                    analysis['bbox_stats']['bbox_centers'].append((center_x, center_y))
        except:
            continue
    
    # Calculate averages
    if objects_per_image:
        analysis['bbox_stats']['avg_objects_per_image'] = np.mean(objects_per_image)
        analysis['bbox_stats']['objects_per_image_dist'] = {
            'min': min(objects_per_image),
            'max': max(objects_per_image),
            'mean': np.mean(objects_per_image),
            'std': np.std(objects_per_image)
        }
    
    return analysis

def _generate_overall_stats(analysis_results, dataset_info):
    """Generate overall dataset statistics"""
    total_images = sum(result['image_count'] for result in analysis_results.values())
    total_objects = sum(result['bbox_stats']['total_objects'] for result in analysis_results.values())
    
    # Combine class distributions
    overall_class_dist = defaultdict(int)
    for split_result in analysis_results.values():
        for class_name, count in split_result['class_distribution'].items():
            overall_class_dist[class_name] += count
    
    # Calculate split ratios
    split_ratios = {}
    for split, result in analysis_results.items():
        split_ratios[split] = result['image_count'] / total_images if total_images > 0 else 0
    
    return {
        'total_images': total_images,
        'total_objects': total_objects,
        'avg_objects_per_image': total_objects / total_images if total_images > 0 else 0,
        'split_ratios': split_ratios,
        'class_distribution': dict(overall_class_dist),
        'most_common_classes': dict(Counter(overall_class_dist).most_common(10)),
        'class_balance_score': _calculate_class_balance_score(overall_class_dist)
    }

def _calculate_class_balance_score(class_dist):
    """Calculate class balance score (0-1, 1 = perfectly balanced)"""
    if not class_dist:
        return 0
    
    counts = list(class_dist.values())
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    
    # Coefficient of variation (lower = more balanced)
    cv = std_count / mean_count if mean_count > 0 else float('inf')
    
    # Convert to 0-1 score (1 = perfectly balanced)
    balance_score = 1 / (1 + cv)
    return round(balance_score, 3)

def _calculate_quality_metrics(analysis_results):
    """Calculate overall quality metrics"""
    metrics = {}
    
    # Data consistency
    for split, result in analysis_results.items():
        image_count = result['image_count']
        label_count = result['label_count']
        consistency = label_count / image_count if image_count > 0 else 0
        metrics[f'{split}_consistency'] = round(consistency, 3)
    
    # Overall quality score
    avg_consistency = np.mean([metrics[f'{split}_consistency'] for split in analysis_results.keys()])
    
    # Check if we have reasonable split sizes
    total_images = sum(result['image_count'] for result in analysis_results.values())
    split_quality = 1.0 if total_images > 100 else total_images / 100
    
    overall_quality = (avg_consistency + split_quality) / 2
    metrics['overall_quality_score'] = round(overall_quality, 3)
    
    return metrics

def _save_analysis_report(report, output_path):
    """Save analysis report to JSON file"""
    os.makedirs(output_path, exist_ok=True)
    report_path = os.path.join(output_path, "dataset_analysis_report.json")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Clean report for JSON
    clean_report = json.loads(json.dumps(report, default=convert_numpy))
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(clean_report, f, indent=2, ensure_ascii=False)
    
    return report_path

def _print_analysis_summary(report):
    """Print analysis summary"""
    print("\n" + "=" * 50)
    print("Dataset Analysis Summary")
    print("=" * 50)
    
    overall = report['overall_stats']
    quality = report['quality_metrics']
    
    print(f"Total Images: {overall['total_images']}")
    print(f"Total Objects: {overall['total_objects']}")
    print(f"Avg Objects/Image: {overall['avg_objects_per_image']:.1f}")
    print(f"Total Classes: {len(overall['class_distribution'])}")
    
    print(f"\nSplit Distribution:")
    for split, ratio in overall['split_ratios'].items():
        count = report['split_analysis'][split]['image_count']
        print(f"  {split.capitalize()}: {count} ({ratio:.1%})")
    
    print(f"\nTop 5 Classes:")
    for class_name, count in list(overall['most_common_classes'].items())[:5]:
        print(f"  {class_name}: {count}")
    
    print(f"\nQuality Metrics:")
    print(f"  Class Balance Score: {overall['class_balance_score']}")
    print(f"  Overall Quality Score: {quality['overall_quality_score']}")
    
    for split in ['train', 'val', 'test']:
        if f'{split}_consistency' in quality:
            print(f"  {split.capitalize()} Consistency: {quality[f'{split}_consistency']}")

def generate_dataset_report(dataset_path, output_path=None):
    """Generate comprehensive dataset analysis report"""
    print("=" * 50)
    print("Dataset Analysis and Quality Report Generation")
    print("=" * 50)
    
    report = analyze_yolo_dataset(dataset_path, output_path)
    
    print(f"\nDataset Analysis Complete!")
    if output_path:
        print(f"Report saved to: {output_path}")
    
    return report

def quick_analyze(dataset_path):
    """Quick analysis without saving report"""
    return analyze_yolo_dataset(dataset_path, output_path=None)

def validate_dataset_balance(dataset_path, min_balance_score=0.5):
    """Validate if dataset is reasonably balanced"""
    report = quick_analyze(dataset_path)
    balance_score = report['overall_stats']['class_balance_score']
    
    if balance_score >= min_balance_score:
        print(f"Dataset balance OK: {balance_score:.3f} >= {min_balance_score}")
        return True
    else:
        print(f"Dataset imbalanced: {balance_score:.3f} < {min_balance_score}")
        print("Consider rebalancing the dataset")
        return False

# Integration function for pipeline
def run_dataset_analysis(dataset_path, output_path):
    """Run complete dataset analysis pipeline"""
    print("=" * 50)
    print("Dataset Analysis and Quality Assessment")
    print("=" * 50)
    
    report = generate_dataset_report(dataset_path, output_path)
    
    # Additional validations
    balance_ok = validate_dataset_balance(dataset_path)
    
    analysis_summary = {
        'total_images': report['overall_stats']['total_images'],
        'total_classes': len(report['overall_stats']['class_distribution']),
        'class_balance_score': report['overall_stats']['class_balance_score'],
        'overall_quality_score': report['quality_metrics']['overall_quality_score'],
        'balance_validation': balance_ok,
        'report_path': os.path.join(output_path, "dataset_analysis_report.json") if output_path else None
    }
    
    return analysis_summary