"""
설정 기반 데이터셋 생성 실행 스크립트

이 스크립트를 사용하면 config.py에 정의된 설정으로 
쉽게 데이터셋을 생성할 수 있습니다.

사용법:
    python run_dataset.py                    # 기본 설정 (development)
    python run_dataset.py development        # 개발용 설정
    python run_dataset.py prototype          # 프로토타입용 설정
    python run_dataset.py --list             # 모든 설정 목록 보기
    python run_dataset.py --help             # 도움말
"""

import sys
import argparse
from pathlib import Path

# 현재 디렉토리의 모듈들 import
from pruning_dataset.modules.dataset_splitter import create_small_dataset
from pruning_dataset.modules.config import get_config, list_all_configs, get_recommended_config, validate_config, DATASET_CONFIGS

def create_dataset_with_config(config_name, source_path="./data"):
    """
    설정 이름으로 데이터셋 생성
    
    Args:
        config_name (str): 설정 이름
        source_path (str): 원본 데이터 경로
    """
    print(f"🔧 '{config_name}' 설정으로 데이터셋 생성 시작...")
    print("=" * 60)
    
    # 설정 가져오기
    config = get_config(config_name)
    
    # 설정 정보 출력
    print(f"📋 설정 정보:")
    print(f"  🏋️ 훈련 데이터: {config['train_size']}개")
    print(f"  🧪 테스트 데이터: {config['test_size']}개") 
    print(f"  🎯 샘플링 전략: {config['sampling_strategy']}")
    print(f"  📁 출력 경로: {config['output_path']}")
    if 'description' in config:
        print(f"  📝 설명: {config['description']}")
    print()
    
    # 설정 유효성 검사
    try:
        validate_config(config)
    except ValueError as e:
        print(f"❌ 설정 오류: {e}")
        return False
    
    # 데이터셋 생성
    try:
        create_small_dataset(
            source_path=source_path,
            output_path=config['output_path'],
            train_size=config['train_size'],
            test_size=config['test_size'],
            sampling_strategy=config['sampling_strategy']
        )
        
        print(f"\n🎉 '{config_name}' 데이터셋 생성 완료!")
        print(f"📁 생성 위치: {config['output_path']}")
        return True
        
    except Exception as e:
        print(f"❌ 데이터셋 생성 실패: {e}")
        return False

def create_multiple_datasets(config_names, source_path="./data"):
    """
    여러 설정으로 데이터셋들 생성
    
    Args:
        config_names (list): 설정 이름 리스트
        source_path (str): 원본 데이터 경로
    """
    print(f"🏗️ {len(config_names)}개의 데이터셋 생성 시작...")
    
    results = []
    for i, config_name in enumerate(config_names, 1):
        print(f"\n{'='*60}")
        print(f"📊 진행률: {i}/{len(config_names)} - '{config_name}' 처리 중...")
        
        success = create_dataset_with_config(config_name, source_path)
        results.append((config_name, success))
    
    # 최종 결과 요약
    print(f"\n{'='*60}")
    print("📋 생성 결과 요약:")
    
    successful = [name for name, success in results if success]
    failed = [name for name, success in results if not success]
    
    if successful:
        print(f"✅ 성공 ({len(successful)}개):")
        for name in successful:
            config = get_config(name)
            print(f"  📁 {name}: {config['output_path']}")
    
    if failed:
        print(f"❌ 실패 ({len(failed)}개):")
        for name in failed:
            print(f"  ⚠️ {name}")
    
    return results

def interactive_mode():
    """대화식 모드로 설정 선택"""
    print("🤖 대화식 데이터셋 생성 모드")
    print("=" * 40)
    
    # 목적 선택
    print("1️⃣ 데이터셋 사용 목적을 선택하세요:")
    purposes = {
        '1': 'quick_test',
        '2': 'development', 
        '3': 'experiment',
        '4': 'production',
        '5': 'demo',
        '6': 'analysis'
    }
    
    for key, purpose in purposes.items():
        print(f"  {key}. {purpose}")
    
    choice = input("\n선택 (1-6): ").strip()
    
    if choice in purposes:
        purpose = purposes[choice]
        recommended = get_recommended_config(purpose)
        
        # 추천 설정 사용할지 확인
        use_recommended = input(f"\n추천 설정 '{recommended}' 를 사용하시겠습니까? (y/n): ").strip().lower()
        
        if use_recommended == 'y':
            return create_dataset_with_config(recommended)
        else:
            print("\n사용 가능한 모든 설정:")
            list_all_configs()
            
            config_name = input("\n사용할 설정 이름을 입력하세요: ").strip()
            return create_dataset_with_config(config_name)
    else:
        print("❌ 잘못된 선택입니다.")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='설정 기반 데이터셋 생성 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_dataset.py                     # 기본 설정 (development)
  python run_dataset.py prototype           # 프로토타입용 데이터셋
  python run_dataset.py experiment          # 실험용 데이터셋
  python run_dataset.py --multiple prototype development experiment
  python run_dataset.py --list              # 모든 설정 목록
  python run_dataset.py --interactive       # 대화식 모드
        """
    )
    
    # 위치 인수 (설정 이름)
    parser.add_argument('config', nargs='?', default='development',
                       help='사용할 설정 이름 (기본값: development)')
    
    # 선택적 인수들
    parser.add_argument('--list', action='store_true',
                       help='모든 사용 가능한 설정 목록 보기')
    
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='대화식 모드로 실행')
    
    parser.add_argument('--multiple', nargs='+', 
                       help='여러 설정으로 한번에 생성 (예: --multiple prototype development)')
    
    parser.add_argument('--source', default='./data',
                       help='원본 데이터 경로 (기본값: ./data)')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='실제 생성 없이 설정만 확인')
    
    args = parser.parse_args()
    
    # 설정 목록 보기
    if args.list:
        list_all_configs()
        return
    
    # 대화식 모드
    if args.interactive:
        interactive_mode()
        return
    
    # 여러 설정 처리
    if args.multiple:
        if args.dry_run:
            print("🔍 Dry-run 모드: 설정만 확인합니다.")
            for config_name in args.multiple:
                config = get_config(config_name)
                print(f"\n📋 {config_name}:")
                print(f"  Train: {config['train_size']}, Test: {config['test_size']}")
                print(f"  Strategy: {config['sampling_strategy']}")
                print(f"  Output: {config['output_path']}")
        else:
            create_multiple_datasets(args.multiple, args.source)
        return
    
    # 단일 설정 처리
    if args.dry_run:
        print("🔍 Dry-run 모드: 설정만 확인합니다.")
        config = get_config(args.config)
        print(f"\n📋 {args.config}:")
        print(f"  Train: {config['train_size']}, Test: {config['test_size']}")
        print(f"  Strategy: {config['sampling_strategy']}")
        print(f"  Output: {config['output_path']}")
    else:
        create_dataset_with_config(args.config, args.source)

if __name__ == "__main__":
    main()