import os
from datetime import datetime

from modules.validation import validate_dataset
from modules.preprocessing import run_preprocessing
from modules.annotation_converter import run_conversion
from modules.dataset_split import run_dataset_split
from modules.data_analyzer import run_dataset_analysis, generate_dataset_report
from modules.data_packager import run_final_packaging
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
    
    # 4단계 추가
    final_dataset_path = os.path.join(data_path, "final_dataset")
    split_result = run_dataset_split(preprocessed_path, yolo_path, final_dataset_path)

    # 5단계 추가
    reports_path = os.path.join(data_path, "reports")
    analysis_result = run_dataset_analysis(final_dataset_path, reports_path)

    # Stage 6: Final Packaging
    delivery_path = os.path.join(data_path, "delivery")
    packaging_result = run_final_packaging(
        final_dataset_path, 
        reports_path, 
        delivery_path,
        f"pill_dataset_{datetime.now().strftime('%Y%m%d')}"
    )

    # Final Summary
    print("\n" + "🎉" * 20)
    print("COMPLETE PIPELINE FINISHED!")
    print("🎉" * 20)
    print(f"📁 Final Package: {packaging_result['archive_path']}")
    print(f"📊 Quality Score: {analysis_result['overall_quality_score']}")
    print(f"📦 Package Size: {packaging_result['package_size_mb']:.1f} MB")
    print(f"✅ Ready for Delivery: {packaging_result['ready_for_delivery']}")
    
    return {
        'validation': validation_result,
        'preprocessing': {'processed': processed, 'failed': failed},
        'conversion': conversion_result,
        'split': split_result,
        'analysis': analysis_result,
        'packaging': packaging_result
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
    
    # 보고서 저장
    report = generate_dataset_report(
        "./data/prototype_data/final_dataset",
        "./data/prototype_data/reports"
    )

    # Example usage:
    DATA_PATH = "./data/prototype_data"
    result = run_full_pipeline(DATA_PATH)