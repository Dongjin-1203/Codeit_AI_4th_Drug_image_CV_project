import os
import sys
from datetime import datetime

# Import small dataset creation functions
from pruning_dataset.modules.dataset_splitter import create_small_dataset
from pruning_dataset.modules.config import get_config, list_all_configs

# Import full processing pipeline functions
from data_preprocess.modules.validation import validate_dataset
from data_preprocess.modules.preprocessing import run_preprocessing
from data_preprocess.modules.annotation_converter import run_conversion
from data_preprocess.modules.dataset_split import run_dataset_split
from data_preprocess.modules.data_analyzer import run_dataset_analysis
from data_preprocess.modules.data_packager import run_final_packaging

# =============================================================================
# Integrated Pipeline Functions
# =============================================================================

def run_complete_pipeline(config_name="development", target_size=1280):
    """
    Complete pipeline: Create small dataset -> Process -> Package for delivery
    
    Args:
        config_name: Dataset size configuration (prototype, development, experiment, etc.)
        target_size: Image processing target size (default 1280)
    """
    print("=" * 60)
    print("COMPLETE DATA PIPELINE STARTING")
    print("=" * 60)
    
    # Stage 0: Create small dataset
    print("\nStage 0: Creating small dataset...")
    config = get_config(config_name)
    create_small_dataset(
        source_path="./data",
        output_path=config['output_path'],
        train_size=config['train_size'],
        test_size=config['test_size'],
        sampling_strategy=config['sampling_strategy']
    )
    
    small_dataset_path = config['output_path']
    print(f"Small dataset created at: {small_dataset_path}")
    
    # Stage 1-6: Full processing pipeline
    print(f"\nStarting full processing pipeline on: {small_dataset_path}")
    
    # Stage 1: Validation
    print("\nStage 1: Data validation...")
    validation_result = validate_dataset(small_dataset_path)
    
    # Stage 2: Preprocessing
    print("\nStage 2: Image preprocessing...")
    preprocessed_path = os.path.join(small_dataset_path, "preprocessed")
    processed, failed = run_preprocessing(small_dataset_path, preprocessed_path, target_size)
    
    # Stage 3: YOLO Conversion
    print("\nStage 3: COCO to YOLO conversion...")
    yolo_path = os.path.join(small_dataset_path, "yolo_labels")
    conversion_result = run_conversion(preprocessed_path, yolo_path)
    
    # Stage 4: Dataset Splitting
    print("\nStage 4: Dataset splitting...")
    final_dataset_path = os.path.join(small_dataset_path, "final_dataset")
    split_result = run_dataset_split(preprocessed_path, yolo_path, final_dataset_path)
    
    # Stage 5: Analysis
    print("\nStage 5: Dataset analysis...")
    reports_path = os.path.join(small_dataset_path, "reports")
    analysis_result = run_dataset_analysis(final_dataset_path, reports_path)
    
    # Stage 6: Final Packaging
    print("\nStage 6: Final packaging...")
    delivery_path = os.path.join(small_dataset_path, "delivery")
    packaging_result = run_final_packaging(
        final_dataset_path, 
        reports_path, 
        delivery_path,
        f"pill_dataset_{config_name}_{datetime.now().strftime('%Y%m%d')}"
    )
    
    # Final Summary
    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE FINISHED!")
    print("=" * 60)
    print(f"Dataset Config: {config_name}")
    print(f"Source Dataset: {small_dataset_path}")
    print(f"Final Package: {packaging_result['archive_path']}")
    print(f"Quality Score: {analysis_result['overall_quality_score']}")
    print(f"Package Size: {packaging_result['package_size_mb']:.1f} MB")
    print(f"Ready for Delivery: {packaging_result['ready_for_delivery']}")
    
    return {
        'config_used': config_name,
        'small_dataset_path': small_dataset_path,
        'validation': validation_result,
        'preprocessing': {'processed': processed, 'failed': failed},
        'conversion': conversion_result,
        'split': split_result,
        'analysis': analysis_result,
        'packaging': packaging_result
    }

def run_custom_pipeline(train_size, test_size, output_name="custom", strategy="balanced", target_size=1280):
    """
    Custom-sized complete pipeline
    
    Args:
        train_size: Number of training images
        test_size: Number of test images
        output_name: Output directory name
        strategy: Sampling strategy (balanced, quality, random)
        target_size: Image processing target size
    """
    print("=" * 60)
    print("CUSTOM PIPELINE STARTING")
    print("=" * 60)
    print(f"Train: {train_size}, Test: {test_size}, Strategy: {strategy}")
    
    # Create custom dataset
    output_path = f"./data/{output_name}_data"
    create_small_dataset(
        source_path="./data",
        output_path=output_path,
        train_size=train_size,
        test_size=test_size,
        sampling_strategy=strategy
    )
    
    # Run processing pipeline on custom dataset
    return run_processing_pipeline_only(output_path, target_size)

def run_processing_pipeline_only(data_path, target_size=1280):
    """
    Run only the processing pipeline (stages 1-6) on existing dataset
    
    Args:
        data_path: Path to existing dataset
        target_size: Image processing target size
    """
    print("=" * 60)
    print("PROCESSING PIPELINE STARTING")
    print("=" * 60)
    print(f"Processing dataset at: {data_path}")
    
    # Stage 1: Validation
    validation_result = validate_dataset(data_path)
    
    # Stage 2: Preprocessing
    preprocessed_path = os.path.join(data_path, "preprocessed")
    processed, failed = run_preprocessing(data_path, preprocessed_path, target_size)
    
    # Stage 3: YOLO Conversion
    yolo_path = os.path.join(data_path, "yolo_labels")
    conversion_result = run_conversion(preprocessed_path, yolo_path)
    
    # Stage 4: Dataset Splitting
    final_dataset_path = os.path.join(data_path, "final_dataset")
    split_result = run_dataset_split(preprocessed_path, yolo_path, final_dataset_path)
    
    # Stage 5: Analysis
    reports_path = os.path.join(data_path, "reports")
    analysis_result = run_dataset_analysis(final_dataset_path, reports_path)
    
    # Stage 6: Final Packaging
    delivery_path = os.path.join(data_path, "delivery")
    packaging_result = run_final_packaging(
        final_dataset_path, 
        reports_path, 
        delivery_path,
        f"pill_dataset_processed_{datetime.now().strftime('%Y%m%d')}"
    )
    
    print("\n" + "=" * 60)
    print("PROCESSING PIPELINE FINISHED!")
    print("=" * 60)
    print(f"Final Package: {packaging_result['archive_path']}")
    
    return {
        'validation': validation_result,
        'preprocessing': {'processed': processed, 'failed': failed},
        'conversion': conversion_result,
        'split': split_result,
        'analysis': analysis_result,
        'packaging': packaging_result
    }

# =============================================================================
# Small Dataset Creation Functions (from pruning_main.py)
# =============================================================================

def create_by_config(config_name):
    """Create dataset using predefined config"""
    config = get_config(config_name)
    create_small_dataset(
        source_path="./data",
        output_path=config['output_path'],
        train_size=config['train_size'],
        test_size=config['test_size'],
        sampling_strategy=config['sampling_strategy']
    )
    print(f"Dataset ready at {config['output_path']}")
    return config['output_path']

def create_small_datasets_batch(config_names):
    """Create multiple small datasets"""
    print(f"Creating {len(config_names)} datasets...")
    created_paths = []
    
    for config_name in config_names:
        print(f"\nCreating {config_name}...")
        path = create_by_config(config_name)
        created_paths.append((config_name, path))
    
    print(f"\nBatch creation complete! Created {len(created_paths)} datasets.")
    return created_paths

# =============================================================================
# Quick Pipeline Functions
# =============================================================================

def quick_prototype_pipeline():
    """빠른 프로토타입: 소규모 데이터셋 + 전체 처리"""
    return run_complete_pipeline("prototype")

def quick_development_pipeline():
    """개발용 파이프라인: 중간 크기 데이터셋 + 전체 처리"""
    return run_complete_pipeline("development")

def quick_experiment_pipeline():
    """실험용 파이프라인: 대용량 데이터셋 + 전체 처리"""
    return run_complete_pipeline("experiment")

def quick_demo_pipeline():
    """데모용 파이프라인: 고품질 데이터셋 + 전체 처리"""
    return run_complete_pipeline("demo")

# =============================================================================
# Interactive Menu
# =============================================================================

def show_pipeline_menu():
    """Interactive menu for pipeline operations"""
    print("\n데이터 파이프라인 메뉴")
    print("=" * 40)
    
    options = {
        '1': ('완전 프로토타입 파이프라인', quick_prototype_pipeline),
        '2': ('완전 개발용 파이프라인', quick_development_pipeline),
        '3': ('완전 실험용 파이프라인', quick_experiment_pipeline),
        '4': ('완전 데모용 파이프라인', quick_demo_pipeline),
        '5': ('커스텀 완전 파이프라인', _custom_complete_menu),
        '6': ('소규모 데이터셋만 생성', _dataset_only_menu),
        '7': ('기존 데이터셋 처리', _process_existing_menu),
        '8': ('사용 가능한 설정 목록', list_all_configs),
        '9': ('일괄 데이터셋 생성', _batch_create_menu)
    }
    
    for key, (desc, _) in options.items():
        print(f"  {key}. {desc}")
    
    choice = input("\nSelect (1-9): ").strip()
    
    if choice in options:
        _, func = options[choice]
        if func:
            try:
                result = func()
                print(f"\nOperation completed successfully!")
                if isinstance(result, dict) and 'packaging' in result:
                    print(f"Final package: {result['packaging']['archive_path']}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        print("Invalid choice")

def _custom_complete_menu():
    """Custom complete pipeline menu"""
    try:
        train_size = int(input("Train size: "))
        test_size = int(input("Test size: "))
        output_name = input("Output name (default: custom): ").strip() or "custom"
        
        strategies = ["balanced", "quality", "random"]
        print(f"Strategies: {', '.join(strategies)}")
        strategy = input("Strategy (default: balanced): ").strip() or "balanced"
        
        if strategy in strategies:
            return run_custom_pipeline(train_size, test_size, output_name, strategy)
        else:
            print("Invalid strategy")
    except ValueError:
        print("Invalid input")

def _dataset_only_menu():
    """Dataset creation only menu"""
    list_all_configs()
    config_name = input("\nSelect config name: ").strip()
    return create_by_config(config_name)

def _process_existing_menu():
    """Process existing dataset menu"""
    data_path = input("Enter dataset path: ").strip()
    if os.path.exists(data_path):
        return run_processing_pipeline_only(data_path)
    else:
        print("Dataset path does not exist")

def _batch_create_menu():
    """Batch dataset creation menu"""
    print("Available configs:")
    list_all_configs()
    
    configs_input = input("\nEnter config names (comma-separated): ").strip()
    config_names = [name.strip() for name in configs_input.split(",")]
    
    return create_small_datasets_batch(config_names)

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main CLI entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        # Complete pipeline commands
        if command == "complete":
            config_name = sys.argv[2] if len(sys.argv) > 2 else "development"
            return run_complete_pipeline(config_name)
        
        # Custom pipeline
        elif command == "custom":
            if len(sys.argv) >= 4:
                try:
                    train_size = int(sys.argv[2])
                    test_size = int(sys.argv[3])
                    output_name = sys.argv[4] if len(sys.argv) > 4 else "custom"
                    strategy = sys.argv[5] if len(sys.argv) > 5 else "balanced"
                    return run_custom_pipeline(train_size, test_size, output_name, strategy)
                except ValueError:
                    print("사용법: python pipeline.py custom <훈련수> <테스트수> [출력이름] [전략]")
            else:
                print("사용법: python pipeline.py custom <훈련수> <테스트수> [출력이름] [전략]")
        
        # Process existing dataset
        elif command == "process":
            if len(sys.argv) > 2:
                data_path = sys.argv[2]
                return run_processing_pipeline_only(data_path)
            else:
                print("사용법: python pipeline.py process <데이터셋경로>")
        
        # Create dataset only
        elif command == "create":
            if len(sys.argv) > 2:
                config_name = sys.argv[2]
                return create_by_config(config_name)
            else:
                print("사용법: python pipeline.py create <설정이름>")
        
        # Quick commands
        elif command == "prototype":
            return quick_prototype_pipeline()
        elif command == "development":
            return quick_development_pipeline()
        elif command == "experiment":
            return quick_experiment_pipeline()
        elif command == "demo":
            return quick_demo_pipeline()
        
        # List configs
        elif command == "list":
            return list_all_configs()
        
        # Show menu
        elif command == "menu":
            return show_pipeline_menu()
        
        # Help
        elif command in ["help", "-h", "--help"]:
            print_help()
        
        else:
            print(f"알 수 없는 명령어: {command}")
            print("사용법 정보를 보려면 'python pipeline.py help'를 입력하세요")
    
    else:
        # No arguments - show menu
        show_pipeline_menu()

def print_help():
    """Print help information"""
    help_text = """
데이터 파이프라인 CLI 도구

사용법:
    python pipeline.py <명령어> [인수]

명령어:
    complete [설정]             완전 파이프라인 실행 (데이터셋 생성 + 처리)
                               기본 설정: development
                               사용 가능: prototype, development, experiment, demo
    
    custom <훈련수> <테스트수> [이름] [전략]
                               지정된 크기로 커스텀 파이프라인 실행
                               예시: python pipeline.py custom 200 100 my_data balanced
    
    process <데이터셋경로>       기존 데이터셋 처리 (1-6단계만)
    
    create <설정>              소규모 데이터셋만 생성 (처리 없음)
                              예시: python pipeline.py create prototype
    
    prototype                  빠른 프로토타입 파이프라인 (50/25 이미지)
    development                빠른 개발용 파이프라인 (150/75 이미지)
    experiment                 빠른 실험용 파이프라인 (300/150 이미지)
    demo                       빠른 데모용 파이프라인 (100/50 이미지)
    
    list                       사용 가능한 모든 설정 목록 보기
    menu                       대화형 메뉴 표시
    help                       이 도움말 메시지 표시

사용 예시:
    python pipeline.py complete prototype
    python pipeline.py custom 100 50 test_data quality
    python pipeline.py process ./data/my_dataset
    python pipeline.py create development
    python pipeline.py prototype

대화형 모드를 사용하려면 인수 없이 실행:
    python pipeline.py
"""
    print(help_text)

if __name__ == "__main__":
    main()