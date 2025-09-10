# 📌 [코드잇 스프린트_AI_4기] 초급 팀프로젝트: 경구약제 이미지 객체 인식 모델
---
코드잇 스프린트 초급 팀프로젝트이다. 이번 프로젝트의 목표는 사진 속에 있는 최대 4개의 알약의 이름(클래스)과 위치(바운딩 박스)를 검출하는 것이다. 또한 하이퍼파라미터 튜닝 등을 통해 최고 성능의 모델을 개발하는 것이 목표이다.
팀은 5인 1팀으로 Project Manager / Data Engineer / Model Architect + Experimentation Lead 로 구성되어있다.
**모델 점수와 상관없이 실무의 팀 개발을 체험하는 과정으로 좋은 인맥 형성, 소프트 스킬 향상, 최선의 팀 결과물 완성이 코드잇에서 말하는 목표이다.**

## 프로젝트 기간: 25.09.09 15:00 ~ 25.09.25 23:50

## kaggle 및 데이터셋 링크
[프로젝트 자료 링크](https://www.kaggle.com/competitions/ai04-level1-project/data)

## 개인 역할

|역할|담당자|업무|
|----|-----|-----|
|Project Manager|신승목|프로젝트 일정관리, 진행상태 확인 및 종합. 최종 보고서 작성, 기타 부족한 부분 지원|
|Data Enginner|지동진|데이터 파이프라인 구축, 데이터 EDA 시행, 파이프라인 자동화|
|Model Architect|이재영|Object Detection 관련 모델 구축|
|Model Architect + Experimentation Sub|남경민|Model Architect, Experimentation Lead 보조|
|Experimentation Lead|이솔형|모델 평가 및 EDA 결과 바탕 모델 성능 개선을 위한 다양한 실험 진행|

---

## 팀프로젝트 수칙
### 1. 데이터 사용 규칙
- 제공된 데이터셋 외 외부 데이터 사용 가능
### 2. 모델 및 코드 제출
- 제출 파일 형식을 준수해 주세요.
- 모델 및 결과물의 재현 가능성을 확보해 주세요.
### 3. 평가 기준 및 리더보드 운영
- 평가 지표: mAP(mean Average Precision)
- 1일 최대 제출 횟수: 5회
- 리더보드는 Public / Private Score로 운영합니다 (최종 순위: Private Score 기준)
---

## 📂 폴더 구성
```
2025-HEALTH-VISION/
├── data/                        # 실제 데이터는 GitHub에 포함되지 않으며,
│   └── data.txt                 # Google Drive 내 데이터 공유 링크가 담긴 텍스트 파일만 존재
├── data_preprocess/             # 데이터 전처리 및 자동화 파이프라인 관련 파일 업로드,
│   ├── data_preprocess.txt      # 디렉토리 개설 목적이 기술 되어있다.
│   └── data_preprocess.py       # 데이터 전처리 자동화 코드 파일
├── notebooks/                   # Jupyter 노트북
│   ├── data_EDA.ipynb           # 데이터 EDA 보고서
│   └── data_pipeline.ipynb      # 데이터 전처리 관련 코드 작성
├── LICENSE                      # 라이센스
├── README.md                    # 프로젝트 문서
├── git_clone.ipynb              # Git clone 실습 코드 
```

---
## 🔧 Git 관련 핵심사항
코랩에서 작업하시고 작업물 올리실때 다음 절차대로 하시면 되겠습니다.
### 1. 작업 시작(코랩)
```
# 1. 기존 리포지토리 폴더로 이동
import os
os.chdir('/content/Codeit_AI_4th_Drug_image_CV_project')

# 2. 최신 변경사항 가져오기
!git pull origin main

# 3. 현재 상태 확인
!git status
!git log --oneline -3
```
### 2. 작업 중
```
# 파일 수정, 코드 작성...
# 중간 저장 (로컬 커밋)
!git add .
!git commit -m "작업 진행 중 - 중간 저장"
```
### 3. 작업 완료
```
# 1. 최종 커밋
!git add .
!git commit -m "실습 내용 추가 및 완료"

# 2. 혹시 다른 팀원이 push했는지 확인
!git pull origin main

# 3. 최종 push
!git push origin main
```
