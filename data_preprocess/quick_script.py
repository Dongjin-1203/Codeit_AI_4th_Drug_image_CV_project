"""
팀원들을 위한 빠른 실행 스크립트들

각 함수를 직접 호출하거나 스크립트를 실행해서 사용할 수 있습니다.
"""

from dataset_prunig import create_small_dataset
from config import get_config

# =============================================================================
# 🛠️ 헬퍼 함수
# =============================================================================

def _create_with_config(config_name):
    """설정으로 데이터셋 생성 (description 필드 제외)"""
    config = get_config(config_name)
    # create_small_dataset에 필요한 파라미터만 추출
    params = {
        'output_path': config['output_path'],
        'train_size': config['train_size'],
        'test_size': config['test_size'],
        'sampling_strategy': config['sampling_strategy']
    }
    
    create_small_dataset(
        source_path="./data",
        **params
    )

# =============================================================================
# 👥 팀원별 빠른 실행 함수들
# =============================================================================

def quick_prototype():
    """🔬 프로토타입용 - 매우 빠른 테스트 (50/25)"""
    print("🔬 프로토타입용 데이터셋 생성 중...")
    _create_with_config("prototype")
    print("✅ 완료! ./data/prototype_data 폴더 확인")

def quick_development():
    """🔧 개발용 - 코드 테스트 및 디버깅 (150/75)"""
    print("🔧 개발용 데이터셋 생성 중...")
    _create_with_config("development")
    print("✅ 완료! ./data/dev_data 폴더 확인")

def quick_experiment():
    """🧪 실험용 - 모델 테스트 (300/150)"""
    print("🧪 실험용 데이터셋 생성 중...")
    _create_with_config("experiment")
    print("✅ 완료! ./data/exp_data 폴더 확인")

def quick_validation():
    """✅ 검증용 - 성능 평가 (500/250)"""
    print("✅ 검증용 데이터셋 생성 중...")
    _create_with_config("validation")
    print("✅ 완료! ./data/val_data 폴더 확인")

def quick_demo():
    """🎯 데모용 - 발표 및 시연 (100/50)"""
    print("🎯 데모용 데이터셋 생성 중...")
    _create_with_config("demo")
    print("✅ 완료! ./data/demo_data 폴더 확인")

def quick_analysis():
    """📊 분석용 - 데이터 분석 및 시각화 (200/100)"""
    print("📊 분석용 데이터셋 생성 중...")
    _create_with_config("analysis")
    print("✅ 완료! ./data/analysis_data 폴더 확인")

# =============================================================================
# 🎯 목적별 추천 함수들
# =============================================================================

def for_new_team_member():
    """🆕 신규 팀원용 - 개발환경 세팅"""
    print("🆕 신규 팀원을 위한 데이터셋 준비 중...")
    print("프로토타입용과 개발용 데이터셋을 생성합니다.\n")
    
    quick_prototype()
    print()
    quick_development()
    
    print("\n🎉 신규 팀원 데이터셋 준비 완료!")
    print("📁 ./data/prototype_data - 빠른 테스트용")
    print("📁 ./data/dev_data - 일반 개발용")

def for_model_experiment():
    """🔬 모델 실험용 - 여러 크기 데이터셋"""
    print("🔬 모델 실험을 위한 여러 크기 데이터셋 생성 중...")
    
    print("1️⃣ 프로토타입용 (빠른 테스트)")
    quick_prototype()
    
    print("\n2️⃣ 실험용 (메인 실험)")  
    quick_experiment()
    
    print("\n3️⃣ 검증용 (최종 평가)")
    quick_validation()
    
    print("\n🎉 모델 실험용 데이터셋 세트 완료!")

def for_presentation():
    """📊 발표용 - 데모 및 분석용"""
    print("📊 발표용 데이터셋 생성 중...")
    
    print("1️⃣ 데모용 (고품질 샘플)")
    quick_demo()
    
    print("\n2️⃣ 분석용 (시각화)")
    quick_analysis()
    
    print("\n🎉 발표용 데이터셋 준비 완료!")

# =============================================================================
# 🛠️ 커스텀 생성 함수들  
# =============================================================================

def create_custom_size(train_size, test_size, output_name="custom", strategy="balanced"):
    """
    커스텀 크기로 데이터셋 생성
    
    Args:
        train_size (int): 훈련 데이터 수
        test_size (int): 테스트 데이터 수  
        output_name (str): 출력 폴더 이름
        strategy (str): 샘플링 전략
    """
    output_path = f"./data/{output_name}_data"
    
    print(f"🛠️ 커스텀 데이터셋 생성 중...")
    print(f"📊 크기: Train {train_size}, Test {test_size}")
    print(f"🎯 전략: {strategy}")
    print(f"📁 출력: {output_path}")
    
    create_small_dataset(
        source_path="./data",
        output_path=output_path,
        train_size=train_size,
        test_size=test_size,
        sampling_strategy=strategy
    )
    
    print(f"✅ 완료! {output_path} 폴더 확인")

def create_team_standard():
    """📏 팀 표준 데이터셋들 일괄 생성"""
    print("📏 팀 표준 데이터셋 일괄 생성 중...")
    
    datasets = [
        ("프로토타입용", quick_prototype),
        ("개발용", quick_development), 
        ("실험용", quick_experiment),
        ("데모용", quick_demo)
    ]
    
    for name, func in datasets:
        print(f"\n🔄 {name} 생성 중...")
        func()
    
    print("\n🎉 팀 표준 데이터셋 일괄 생성 완료!")
    print("📋 생성된 데이터셋:")
    print("  📁 ./data/prototype_data - 프로토타입용")
    print("  📁 ./data/dev_data - 개발용") 
    print("  📁 ./data/exp_data - 실험용")
    print("  📁 ./data/demo_data - 데모용")

# =============================================================================
# 📱 인터랙티브 메뉴
# =============================================================================

def interactive_menu():
    """간단한 인터랙티브 메뉴"""
    print("🎮 데이터셋 생성 메뉴")
    print("=" * 30)
    
    menu_options = {
        '1': ('프로토타입용 (50/25)', quick_prototype),
        '2': ('개발용 (150/75)', quick_development),
        '3': ('실험용 (300/150)', quick_experiment), 
        '4': ('검증용 (500/250)', quick_validation),
        '5': ('데모용 (100/50)', quick_demo),
        '6': ('분석용 (200/100)', quick_analysis),
        '7': ('신규 팀원용 세트', for_new_team_member),
        '8': ('모델 실험용 세트', for_model_experiment),
        '9': ('발표용 세트', for_presentation),
        '0': ('팀 표준 전체', create_team_standard)
    }
    
    for key, (desc, _) in menu_options.items():
        print(f"  {key}. {desc}")
    
    choice = input("\n선택하세요 (0-9): ").strip()
    
    if choice in menu_options:
        desc, func = menu_options[choice]
        print(f"\n🚀 {desc} 시작...")
        func()
    else:
        print("❌ 잘못된 선택입니다.")

# =============================================================================
# 🎯 메인 실행 부분
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # 명령줄 인수로 함수 직접 호출 가능
    if len(sys.argv) > 1:
        function_name = sys.argv[1]
        
        functions = {
            'prototype': quick_prototype,
            'development': quick_development,
            'experiment': quick_experiment,
            'validation': quick_validation,
            'demo': quick_demo,
            'analysis': quick_analysis,
            'new_member': for_new_team_member,
            'model_exp': for_model_experiment,
            'presentation': for_presentation,
            'team_standard': create_team_standard,
            'menu': interactive_menu
        }
        
        if function_name in functions:
            functions[function_name]()
        else:
            print(f"❌ 알 수 없는 함수: {function_name}")
            print("사용 가능한 함수:", list(functions.keys()))
    else:
        # 기본값: 인터랙티브 메뉴
        interactive_menu()