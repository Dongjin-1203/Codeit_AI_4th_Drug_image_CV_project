"""
데이터셋 생성을 위한 설정 파일

팀원들이 각자 필요에 맞는 데이터셋을 쉽게 생성할 수 있도록
사전 정의된 설정들을 제공합니다.

사용법:
    from config import get_config
    config = get_config("development")
"""

# 데이터셋 설정 정의
DATASET_CONFIGS = {
    # 🔬 프로토타입용 - 매우 빠른 테스트
    "prototype": {
        "train_size": 50,
        "test_size": 25,
        "output_path": "./data/prototype_data",
        "sampling_strategy": "random",
        "description": "매우 빠른 프로토타입 테스트용 (1-2분 실행)"
    },
    
    # 🔧 개발용 - 코드 테스트 및 디버깅
    "development": {
        "train_size": 150,
        "test_size": 75,
        "output_path": "./data/dev_data",
        "sampling_strategy": "balanced",
        "description": "개발 및 디버깅용 (5-10분 실행)"
    },
    
    # 🧪 실험용 - 모델 아키텍처 테스트
    "experiment": {
        "train_size": 300,
        "test_size": 150,
        "output_path": "./data/exp_data",
        "sampling_strategy": "quality",
        "description": "모델 실험 및 하이퍼파라미터 튜닝용 (15-25분 실행)"
    },
    
    # ✅ 검증용 - 성능 평가
    "validation": {
        "train_size": 500,
        "test_size": 250,
        "output_path": "./data/val_data",
        "sampling_strategy": "balanced",
        "description": "모델 성능 검증용 (30-45분 실행)"
    },
    
    # 🎯 데모용 - 발표 및 시연
    "demo": {
        "train_size": 100,
        "test_size": 50,
        "output_path": "./data/demo_data",
        "sampling_strategy": "quality",
        "description": "발표 및 데모용 고품질 샘플"
    },
    
    # 📊 분석용 - 데이터 분석 및 시각화
    "analysis": {
        "train_size": 200,
        "test_size": 100,
        "output_path": "./data/analysis_data",
        "sampling_strategy": "balanced",
        "description": "데이터 분석 및 시각화용"
    }
}

# 개인별 맞춤 설정 (팀원 이름으로 구분)
PERSONAL_CONFIGS = {
    # 예시: 팀원별 설정
    "researcher_a": {
        "train_size": 400,
        "test_size": 200,
        "output_path": "./data/researcher_a_data",
        "sampling_strategy": "balanced",
        "description": "연구원 A 전용 설정"
    },
    
    "researcher_b": {
        "train_size": 250,
        "test_size": 125,
        "output_path": "./data/researcher_b_data", 
        "sampling_strategy": "quality",
        "description": "연구원 B 전용 설정"
    }
}

def get_config(config_name):
    """
    설정 이름으로 설정 정보 가져오기
    
    Args:
        config_name (str): 설정 이름 ('prototype', 'development', 등)
    
    Returns:
        dict: 설정 정보
    """
    # 일반 설정에서 먼저 찾기
    if config_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[config_name]
    
    # 개인 설정에서 찾기
    if config_name in PERSONAL_CONFIGS:
        return PERSONAL_CONFIGS[config_name]
    
    # 설정을 찾을 수 없으면 기본값 반환
    print(f"⚠️ '{config_name}' 설정을 찾을 수 없습니다. 'development' 설정을 사용합니다.")
    return DATASET_CONFIGS["development"]

def list_all_configs():
    """모든 사용 가능한 설정 목록 출력"""
    print("📋 사용 가능한 데이터셋 설정:")
    print("=" * 60)
    
    print("\n🎯 일반 설정:")
    for name, config in DATASET_CONFIGS.items():
        print(f"  {name:12} - {config['description']}")
        print(f"{'':14} Train: {config['train_size']:3d}, Test: {config['test_size']:3d}, Strategy: {config['sampling_strategy']}")
        print()
    
    if PERSONAL_CONFIGS:
        print("👤 개인별 설정:")
        for name, config in PERSONAL_CONFIGS.items():
            print(f"  {name:12} - {config['description']}")
            print(f"{'':14} Train: {config['train_size']:3d}, Test: {config['test_size']:3d}, Strategy: {config['sampling_strategy']}")
            print()

def add_personal_config(name, train_size, test_size, output_path, sampling_strategy="balanced", description=""):
    """
    개인 설정 추가
    
    Args:
        name (str): 설정 이름
        train_size (int): 훈련 데이터 수
        test_size (int): 테스트 데이터 수  
        output_path (str): 출력 경로
        sampling_strategy (str): 샘플링 전략
        description (str): 설명
    """
    PERSONAL_CONFIGS[name] = {
        "train_size": train_size,
        "test_size": test_size,
        "output_path": output_path,
        "sampling_strategy": sampling_strategy,
        "description": description or f"{name} 전용 설정"
    }
    print(f"✅ '{name}' 설정이 추가되었습니다.")

def get_recommended_config(purpose):
    """
    목적에 따른 추천 설정
    
    Args:
        purpose (str): 'quick_test', 'development', 'experiment', 'production'
    
    Returns:
        str: 추천 설정 이름
    """
    recommendations = {
        'quick_test': 'prototype',
        'development': 'development', 
        'experiment': 'experiment',
        'production': 'validation',
        'demo': 'demo',
        'analysis': 'analysis'
    }
    
    recommended = recommendations.get(purpose, 'development')
    print(f"💡 '{purpose}' 목적으로는 '{recommended}' 설정을 추천합니다.")
    return recommended

def validate_config(config):
    """설정 유효성 검사"""
    required_keys = ['train_size', 'test_size', 'output_path', 'sampling_strategy']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"설정에 '{key}' 항목이 없습니다.")
    
    if config['train_size'] <= 0 or config['test_size'] <= 0:
        raise ValueError("train_size와 test_size는 0보다 커야 합니다.")
    
    if config['sampling_strategy'] not in ['balanced', 'quality', 'random']:
        raise ValueError("sampling_strategy는 'balanced', 'quality', 'random' 중 하나여야 합니다.")
    
    return True

# 설정 사용 예시
if __name__ == "__main__":
    # 모든 설정 목록 보기
    list_all_configs()
    
    # 특정 설정 가져오기
    config = get_config("development")
    print(f"\n선택된 설정: {config}")
    
    # 목적별 추천 설정
    get_recommended_config("quick_test")