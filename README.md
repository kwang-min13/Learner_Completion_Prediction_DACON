# 🎓 Learner Completion Prediction

데이콘 경진대회: **학습자 과정 이수 여부 예측**

교육 데이터를 기반으로 학습자가 과정을 **이수할지(completed)** 예측하는 이진 분류 모델을 개발합니다.

---

## 📂 프로젝트 구조

```
Learner_Completion_Prediction
├── train.csv                  # 학습 데이터
├── test.csv                   # 테스트 데이터
├── sample_submission.csv      # 제출 샘플
├── submission.csv             # 최종 제출 파일
└── Learner_modeling.ipynb     # 메인 모델링 노트북
```

---

## 📊 데이터 개요

| 구분 | 수량 |
|------|------|
| 학습 샘플 수 | 748 |
| 원본 피처 수 | 45 |
| 사용 피처 수 (전처리 후) | 38 |
| 타겟 변수 | `completed` (0/1) |

### 주요 피처

| 피처명 | 설명 |
|--------|------|
| `school1` | 학교 코드 |
| `major type` | 전공 유형 |
| `major1_1`, `major1_2` | 세부 전공 |
| `major_data` | 데이터 관련 전공 여부 |
| `job` | 직업 유형 |
| `class1`, `class2` | 기수 정보 |
| `re_registration` | 재등록 여부 |
| `nationality` | 국적 |
| `inflow_route` | 유입 경로 |
| `whyBDA` | 지원 이유 |
| `what_to_gain` | 취득 목표 |
| `completed_semester` | 이수 학기 수 |
| `time_input` | 학습 투입 시간 |
| `certificate_acquisition` | 자격증 취득 현황 |
| `desired_job` | 희망 직무 |
| `incumbents_*` | 재직자 관련 정보 |

---

## ⚙️ 전처리

1. **결측값 90% 초과 컬럼 제거**
   - `class3`, `class4`, `contest_award`, `contest_participation`, `idea_contest`

2. **불필요 컬럼 제거**
   - `ID`, `generation`

3. **결측값 처리**
   - 범주형: `"na"` 로 채움
   - 수치형: `0` 으로 채움

4. **이상값 처리**
   - `completed_semester`: 이상치(`20241.0`, `2020.02`) → 중앙값(median)으로 대체
   - `time_input`: 이상치(`24.0`) → 중앙값으로 대체

5. **범주형 인코딩**
   - 모든 범주형 컬럼 → **LabelEncoder** 로 수치 인코딩

---

## 🤖 모델링

### 사용 라이브러리

```python
catboost==1.2.10
scikit-learn
```

### 모델

- **CatBoostClassifier**

### 검증 전략

- **StratifiedKFold** (5-Fold 교차 검증)
- 평가 지표: **F1-Score**, Accuracy

---

## 🔧 주요 유틸리티 함수

| 함수 | 설명 |
|------|------|
| `major_group(x)` | 전공명 → 그룹 레이블 변환 |
| `merge_rare_train_test(...)` | 희귀 카테고리 병합 |
| `count_by_comma(s)` | 쉼표 구분 항목 수 카운트 |
| `yes_to_int(s)` | Yes/No 이진 변환 |
| `valid_count(df, cols)` | 유효 값 수 카운트 |
| `summary(df)` | 데이터 요약 (타입, 결측값, 고윳값 수) |
| `plot_one_box(df, col)` | 단일 컬럼 박스플롯 시각화 |

---

## 📝 참고

- 플랫폼: [DACON](https://dacon.io)
- 환경: Google Colab (Python 3.12)
