import os
import json
import glob
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter

def create_small_dataset(
    source_path="./data", 
    output_path="./data/small_data",
    train_size=200, 
    test_size=100,
    sampling_strategy="balanced"
):
    """
    소규모 데이터셋 추출 (디렉토리 구조 유지)
    
    Args:
        source_path: 원본 데이터셋 경로
        output_path: 소규모 데이터셋 저장 경로
        train_size: 추출할 훈련 데이터 수
        test_size: 추출할 테스트 데이터 수
        sampling_strategy: "balanced" | "random" | "quality"
    """
    
    print("🔧 소규모 데이터셋 추출 시작...")
    print(f"📊 목표: Train {train_size}개, Test {test_size}개")
    print(f"📁 출력 경로: {output_path}")
    print("=" * 50)
    
    # 출력 디렉토리 생성
    setup_output_directories(output_path)
    
    # 1. 훈련 데이터 추출
    train_pairs = extract_train_data(source_path, train_size, sampling_strategy)
    copy_train_data(train_pairs, source_path, output_path)
    
    # 2. 테스트 데이터 추출
    test_files = extract_test_data(source_path, test_size)
    copy_test_data(test_files, source_path, output_path)
    
    # 3. 결과 확인
    verify_small_dataset(output_path)
    
    print("✅ 소규모 데이터셋 추출 완료!")

def setup_output_directories(output_path):
    """출력 디렉토리 구조 설정"""
    directories = [
        os.path.join(output_path, "train_images"),
        os.path.join(output_path, "test_images"),
        os.path.join(output_path, "train_annotations")
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    
    print("📁 디렉토리 구조 생성 완료")

def extract_train_data(source_path, train_size, sampling_strategy):
    """훈련 데이터 추출 (이미지-어노테이션 매칭)"""
    print("🔍 훈련 데이터 분석 중...")
    
    # 모든 훈련 이미지 파일 찾기
    train_images = glob.glob(os.path.join(source_path, "train_images", "*.png"))
    
    # 이미지-어노테이션 매칭 확인
    valid_pairs = []
    annotation_folders = glob.glob(os.path.join(source_path, "train_annotations", "*"))
    
    for img_path in train_images:
        img_name = os.path.basename(img_path).replace('.png', '')
        
        # 해당하는 JSON 파일 찾기
        json_path = find_matching_annotation(img_name, annotation_folders)
        
        if json_path:
            valid_pairs.append({
                'image_path': img_path,
                'annotation_path': json_path,
                'image_name': img_name
            })
    
    print(f"📊 매칭된 이미지-어노테이션 쌍: {len(valid_pairs)}개")
    
    # 샘플링 전략에 따라 선택
    if sampling_strategy == "balanced":
        selected_pairs = balanced_sampling(valid_pairs, train_size)
    elif sampling_strategy == "quality":
        selected_pairs = quality_sampling(valid_pairs, train_size)
    else:  # random
        selected_pairs = random.sample(valid_pairs, min(train_size, len(valid_pairs)))
    
    print(f"✅ {len(selected_pairs)}개 훈련 샘플 선택 완료")
    return selected_pairs

def find_matching_annotation(img_name, annotation_folders):
    """이미지에 매칭되는 어노테이션 파일 찾기"""
    for folder in annotation_folders:
        # 각 하위 폴더에서 JSON 파일 찾기
        json_files = glob.glob(os.path.join(folder, "*", f"{img_name}.json"))
        if json_files:
            return json_files[0]
    return None

def balanced_sampling(valid_pairs, target_size):
    """알약 코드별 균등 샘플링"""
    print("⚖️ 균등 샘플링 적용 중...")
    
    # 알약 코드별로 그룹핑
    pill_groups = defaultdict(list)
    
    for pair in valid_pairs:
        # 파일명에서 알약 코드 추출 (첫 번째 K- 코드)
        filename = pair['image_name']
        try:
            pill_code = filename.split('-')[1]  # K-003544에서 003544 추출
            pill_groups[pill_code].append(pair)
        except:
            pill_groups['unknown'].append(pair)
    
    print(f"🏷️ 발견된 알약 코드: {len(pill_groups)}개")
    
    # 각 그룹에서 균등하게 샘플링
    samples_per_group = max(1, target_size // len(pill_groups))
    selected_pairs = []
    
    for code, pairs in pill_groups.items():
        sample_count = min(samples_per_group, len(pairs))
        selected = random.sample(pairs, sample_count)
        selected_pairs.extend(selected)
        print(f"  {code}: {sample_count}개 선택")
    
    # 목표 수에 맞게 조정
    if len(selected_pairs) < target_size:
        remaining = target_size - len(selected_pairs)
        remaining_pairs = [p for p in valid_pairs if p not in selected_pairs]
        additional = random.sample(remaining_pairs, min(remaining, len(remaining_pairs)))
        selected_pairs.extend(additional)
    elif len(selected_pairs) > target_size:
        selected_pairs = random.sample(selected_pairs, target_size)
    
    return selected_pairs

def quality_sampling(valid_pairs, target_size):
    """품질 기반 샘플링 (파일 크기 고려)"""
    print("🎯 품질 기반 샘플링 적용 중...")
    
    # 파일 크기 기준으로 정렬
    pairs_with_size = []
    for pair in valid_pairs:
        file_size = os.path.getsize(pair['image_path'])
        pairs_with_size.append((pair, file_size))
    
    # 파일 크기 상위 선택 (큰 파일 = 고품질 가정)
    pairs_with_size.sort(key=lambda x: x[1], reverse=True)
    selected_pairs = [pair for pair, _ in pairs_with_size[:target_size]]
    
    print(f"📈 상위 {len(selected_pairs)}개 고품질 이미지 선택")
    return selected_pairs

def copy_train_data(selected_pairs, source_path, output_path):
    """선택된 훈련 데이터 복사"""
    print("📋 훈련 데이터 복사 중...")
    
    for i, pair in enumerate(selected_pairs):
        # 이미지 복사
        img_src = pair['image_path']
        img_dst = os.path.join(output_path, "train_images", os.path.basename(img_src))
        shutil.copy2(img_src, img_dst)
        
        # 어노테이션 복사 (디렉토리 구조 유지)
        json_src = pair['annotation_path']
        
        # 원본 어노테이션 경로에서 상대 경로 추출
        rel_path = os.path.relpath(json_src, os.path.join(source_path, "train_annotations"))
        json_dst = os.path.join(output_path, "train_annotations", rel_path)
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(json_dst), exist_ok=True)
        shutil.copy2(json_src, json_dst)
        
        if (i + 1) % 50 == 0:
            print(f"  진행률: {i + 1}/{len(selected_pairs)}")
    
    print(f"✅ {len(selected_pairs)}개 훈련 데이터 복사 완료")

def extract_test_data(source_path, test_size):
    """테스트 데이터 추출"""
    print("🧪 테스트 데이터 선택 중...")
    
    test_images = glob.glob(os.path.join(source_path, "test_images", "*.png"))
    selected_files = random.sample(test_images, min(test_size, len(test_images)))
    
    print(f"✅ {len(selected_files)}개 테스트 이미지 선택 완료")
    return selected_files

def copy_test_data(selected_files, source_path, output_path):
    """선택된 테스트 데이터 복사"""
    print("📋 테스트 데이터 복사 중...")
    
    for img_src in selected_files:
        img_dst = os.path.join(output_path, "test_images", os.path.basename(img_src))
        shutil.copy2(img_src, img_dst)
    
    print(f"✅ {len(selected_files)}개 테스트 데이터 복사 완료")

def verify_small_dataset(output_path):
    """추출된 데이터셋 검증"""
    print("\n🔍 데이터셋 검증 중...")
    
    train_imgs = len(glob.glob(os.path.join(output_path, "train_images", "*.png")))
    test_imgs = len(glob.glob(os.path.join(output_path, "test_images", "*.png")))
    annotations = len(glob.glob(os.path.join(output_path, "train_annotations", "*", "*", "*.json")))
    
    print(f"📊 최종 결과:")
    print(f"  🏋️ 훈련 이미지: {train_imgs}개")
    print(f"  🧪 테스트 이미지: {test_imgs}개")
    print(f"  📝 어노테이션: {annotations}개")
    
    # 디렉토리 구조 출력
    print(f"\n📁 생성된 디렉토리 구조:")
    for root, dirs, files in os.walk(output_path):
        level = root.replace(output_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        if files:
            print(f"{subindent}파일 {len(files)}개")

# create_small_dataset 파라미터
"""
기호에 맞게 조정해서 사용하세요!
create_small_dataset 함수의 파라미터를 변경하면 됩니다.

# 1. 기본 사용법 (균등 샘플링)
create_small_dataset(
    source_path="./data",
    output_path="./data/small_data",
    train_size=200,
    test_size=100,
    sampling_strategy="balanced"
)

# 2. 품질 우선 샘플링
create_small_dataset(
    source_path="./data",
    output_path="./data/small_data_quality",
    train_size=150,
    test_size=80,
    sampling_strategy="quality"
)

# 3. 완전 랜덤 샘플링(프로토타입용)
create_small_dataset(
    source_path="./data",
    output_path="./data/small_data_prototype",
    train_size=300,
    test_size=150,
    sampling_strategy="random"
)

# 4. 중간규모 데이터셋 생성(최종 검증용) 
create_small_dataset(
    source_path="./data",
    output_path="./data/small_data_meidium",
    train_size=750,
    test_size=400,
    sampling_strategy="random"
)
"""

def create_quick_small_dataset(train_count=200, test_count=100):
    """소규모 데이터셋 생성기"""
    create_small_dataset(
        source_path="./data",
        output_path="./data/small_data",
        train_size=train_count,
        test_size=test_count,
        sampling_strategy="balanced"
    )
    print(f"🎉 소규모 데이터셋 준비 완료! './data/small_data' 폴더를 확인하세요.")

# 사용법: 그냥 실행하면 됨
# if __name__ == "__main__":
#     create_quick_small_dataset()