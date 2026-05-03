# 학습자 수료 예측 모델링 (Learner Completion Prediction)

## 프로젝트 개요
> **데이콘_**
>
> 이 프로젝트는 BDA(빅데이터 분석) 교육 프로그램 수강생들의 **수료 여부(completed)**를 예측하는 이진 분류 머신러닝 모델을 개발한 것입니다. 수강생들의 인적 정보, 전공, 취업 목표, 수강 이력 등 다양한 피처를 기반으로 CatBoost 모델을 학습시켰습니다.

---

## 데이터 설명

- **학습 데이터**: `train.csv` — 748개 샘플, 45개 피처 (타겟 포함)
- **테스트 데이터**: `test.csv`
- **타겟 변수**: `completed` (수료 여부, 0 또는 1)

### 주요 피처

| 피처 | 타입 | 설명 |
|---|---|---|
| `school1` | int | 학교 코드 (92가지 고유값) |
| `major type` | object | 전공 유형 (3가지) |
| `major1_1`, `major1_2` | object | 세부 전공 |
| `major_data` | bool | 데이터 관련 전공 여부 |
| `job` | object | 재직 상태 (4가지) |
| `class1`, `class2` | int/float | 수업 정보 |
| `re_registration` | object | 재등록 여부 |
| `nationality` | object | 국적 |
| `inflow_route` | object | 유입 경로 |
| `whyBDA` | object | BDA 지원 동기 |
| `what_to_gain` | object | 기대 학습 성과 |
| `desired_career_path` | object | 희망 커리어 경로 |
| `completed_semester` | float | 완료한 학기 수 |
| `time_input` | float | 투입 시간 |
| `desired_job` | object | 희망 직무 |
| ... | ... | (총 38개 피처 사용) |

> **참고**: 결측치 비율이 90% 이상인 컬럼(`class3`, `class4`, `contest_award`, `contest_participation`, `idea_contest`)은 분석에서 제외했습니다.

---

## 파이프라인 요약

```
데이터 로드
    ↓
탐색적 데이터 분석 (EDA)
    ↓
결측치 처리 및 이상값 제거
    ↓
범주형 / 수치형 피처 분리
    ↓
결측치 대체 (범주형: 'na', 수치형: 0)
    ↓
이상값 처리 (completed_semester, time_input)
    ↓
LabelEncoder로 범주형 피처 인코딩
    ↓
StandardScaler / MinMaxScaler로 스케일링
    ↓
CatBoost + StratifiedKFold 학습
    ↓
OOF 임계값 최적화 (F1 Score 기준)
    ↓
최종 예측 및 제출
```

---

## 전처리 상세

### 1. 결측치 처리
- 결측 비율 **90% 이상** 컬럼 일괄 제거
- `ID`, `generation` 컬럼 제거 (식별자/상수값)
- 범주형 컬럼의 결측치 → `'na'` 문자열로 대체
- 수치형 컬럼의 결측치 → `0`으로 대체

### 2. 이상값 처리
- `completed_semester`: `20241.0`, `2020.02` 같은 명백한 입력 오류를 **중위값(median)**으로 대체
- `time_input`: `24.0` (24시간이라는 비현실적 값)을 **중위값**으로 대체

### 3. 범주형 인코딩
- 모든 범주형 컬럼에 대해 `LabelEncoder` 적용
- 학습셋과 테스트셋을 **각각 독립적으로** fit & transform

### 4. 스케일링
- `StandardScaler`와 `MinMaxScaler` 두 가지 방식을 모두 시도
- 두 버전 모두 모델에 투입하여 비교

---

## 모델링

### 알고리즘
- **CatBoost Classifier**
  - 범주형 데이터 처리에 강한 Gradient Boosting 계열 모델
  - 과적합 방지에 유리한 ordered boosting 사용

### 교차 검증
- **StratifiedKFold** (n_splits=5)
  - 클래스 불균형 문제를 고려해 계층적 분할 적용
  - OOF(Out-of-Fold) 예측으로 최종 임계값 탐색

### 임계값 최적화
- 0.01 ~ 0.99 범위에서 **F1 Score 기준** 최적 임계값 탐색
- 최적 결과: `best_f1 ≈ 0.4735`, `best_t ≈ 0.253`

---

## 사용 라이브러리

| 라이브러리 | 용도 |
|---|---|
| `pandas` | 데이터 처리 |
| `numpy` | 수치 연산 |
| `catboost` | CatBoost 분류 모델 |
| `scikit-learn` | 전처리, 교차 검증, 평가 지표 |
| `seaborn` / `matplotlib` | 데이터 시각화 |

---

## 한계점 및 보완할 점

솔직히 말하면, 이 프로젝트를 진행하면서 아쉬움이 남는 부분이 꽤 있었습니다.

### 아쉬웠던 점들

1. **LabelEncoder를 train/test에 따로 적용한 문제**
   가장 신경 쓰이는 부분입니다. 범주형 인코딩을 학습셋과 테스트셋에 각각 독립적으로 `fit`했기 때문에, 같은 범주가 다른 숫자로 인코딩될 수 있습니다. 실제로 서비스나 대회에서 이런 방식은 data leakage와는 다른 문제지만, 범주 매핑의 불일치를 일으킬 수 있어 `fit`은 학습 데이터에만 하고 `transform`을 양쪽에 적용하는 방식이 더 올바릅니다.

2. **피처 엔지니어링 부재**
   `desired_job`, `interested_company`, `expected_domain` 같은 텍스트 형 컬럼들은 단순 LabelEncoding 대신, 키워드 추출이나 임베딩 기반 처리를 했다면 훨씬 더 풍부한 정보를 모델에 전달할 수 있었을 것 같습니다.

3. **클래스 불균형 처리 미흡**
   F1 Score 기준 최적 임계값이 0.253으로 상당히 낮게 나온 건, 아마도 클래스 불균형의 영향이 크다고 봅니다. SMOTE나 클래스 가중치 조정 같은 불균형 처리 기법을 더 적극적으로 활용했으면 어땠을까 싶습니다.

4. **데이터 양의 한계**
   학습 데이터가 748개 샘플이라 꽤 적습니다. 적은 데이터에서는 모델이 일반화하기 어렵고, 교차 검증 결과의 분산도 높아질 수밖에 없었습니다.

---

## 배운 점

이번 프로젝트를 통해 단순히 모델을 돌리는 것 이상의 것을 배웠습니다.

- **전처리의 중요성**: 아무리 좋은 모델을 써도 데이터 품질이 나쁘면 결과도 나쁩니다. `completed_semester`에 `20241.0` 같은 값이 있다는 걸 처음 발견했을 때 꽤 당황했는데, 이런 이상값을 직접 찾고 처리하는 경험이 쌓이면 나중에 새 데이터를 봤을 때 훨씬 빠르게 문제를 캐치할 수 있게 됩니다.

- **OOF 기반 임계값 최적화**: 단순히 0.5를 임계값으로 쓰는 것보다, OOF 예측값을 활용해 F1 Score를 최대화하는 임계값을 직접 탐색하는 방식이 실전에서 얼마나 중요한지 체감했습니다.

- **StratifiedKFold의 중요성**: 클래스 비율을 유지하면서 데이터를 분할하는 것이 단순 KFold보다 안정적인 검증 결과를 준다는 걸 직접 확인했습니다.

- **EDA의 습관화**: 박스플롯, KDE 플롯 등을 통해 각 피처의 분포를 먼저 살펴보는 습관이 전처리 방향을 잡는 데 얼마나 도움이 되는지 다시 한번 느꼈습니다.

---
