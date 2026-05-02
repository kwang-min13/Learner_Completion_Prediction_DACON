### Learner_Completion_Prediction
본 프로젝트는 데이콘(DACON)에서 주최한 '학습자 수료 예측 AI 경진대회'에 참여하며 작성한 베이스라인 모델링 프로젝트입니다. 학습자의 환경, 동기, 행동 데이터를 바탕으로 해당 교육 과정의 최종 수료 여부를 예측(Binary Classification)하는 것을 목표로 합니다.

단순히 모델을 학습시키는 것을 넘어, 현실 데이터의 이상치와 결측치를 논리적으로 처리하고 분류 모델의 평가 지표(F1-score)를 극대화하기 위한 임계값(Threshold) 최적화 과정을 경험할 수 있었습니다.

주요 변수 (Feature)
식별자: ID
타겟 변수: completed (수료 여부, 이진 분류)
배경 정보: school1 , major_type , major_data , job , nationality 등
학습 동기: inflow_route , whyBDA , what_to_gain , hope_for_group 등
학습 이력: class1 ~ class4 , previous_class_3 ~ previous_class_8 등
진로 및 의지: desired_career_path , time_input , desired_job ,
certificate_acquisition 등

### 1. 데이터 전처리 및 탐색 (Data Preprocessing & EDA)
데이터셋의 90% 이상이 범주형(Categorical)으로 이루어져 있으며, 학습자의 의지를 간접적으로 보여주는 다양한 결측치와 이상치가 존재했습니다.

노이즈 및 불필요한 피처 제거:

결측치가 90% 이상인 피처(class3, idea_contest 등)와 예측에 무의미한 식별자(ID, generation)를 모델링 전에 과감히 제거하여 과적합을 방지했습니다.

이상치(Outlier) 보정:

수치형 데이터 탐색(Boxplot 시각화) 중 completed_semester 변수에서 '20241.0', '2020.02'와 같은 명백한 오류 값을 발견했습니다. 이를 전체 데이터의 중앙값(Median)으로 대치하여 분포가 왜곡되지 않도록 안정성을 확보했습니다.

결측치(Missing Value) 대치:

범주형 변수의 빈칸은 na라는 새로운 텍스트로 일괄 대치하여 '응답하지 않음' 자체를 하나의 특성으로 학습하도록 유도했습니다.

수치형 변수의 결측치는 0으로 대치하였습니다.

스케일링 및 시각화:

MinMaxScaler와 StandardScaler를 적용해 보고, 연속형 변수들의 정규화 이후 분포를 KDE Plot을 통해 시각적으로 비교 검증했습니다.

### 2. 모델링 및 검증 전략 (Modeling)
범주형 변수에 강력한 성능을 보이는 CatBoost를 메인 모델로 선정했습니다.

교차 검증 (Stratified K-Fold):

타겟 변수(수료/미수료)의 불균형을 고려하여 Stratified K-Fold (n_splits=5)를 적용했습니다. 이를 통해 모델의 일반화 성능을 높이고 OOF(Out-of-Fold) 예측값을 산출했습니다.

F1-Score 임계값(Threshold) 최적화:

기본 임계값인 0.5에 의존하지 않고, 0.01부터 0.99까지 반복문을 통해 탐색(np.linspace)하며 모델의 F1-score가 가장 높게 산출되는 최적의 Threshold를 직접 도출했습니다.

### 3. 프로젝트 회고 (Retrospective)
## 한계점 및 보완할 점

# Label Encoding 데이터 누수(Data Leakage) 오류: 

범주형 데이터 인코딩 과정에서 Train 데이터와 Test 데이터에 각각 encoder.fit()을 적용했습니다. 이는 Train과 Test의 라벨 맵핑이 달라질 수 있는 심각한 오류입니다. 향후에는 Train 데이터로만 fit을 수행한 후 Test 데이터를 transform 하거나, CatBoost의 내장 범주형 처리 기능(cat_features)을 사용하여 이 문제를 원천적으로 방지해야 합니다.

# Feature Engineering의 부재:

단순히 주어진 데이터를 정제하는 데 그치고, 변수 간의 관계를 활용한 새로운 파생 변수를 생성하지 못했습니다. (예: 진도율 대비 퀴즈 점수 등)

# 스케일링 후 컬럼 드롭 시점:

MinMaxScaler를 적용한 이후에 특정 컬럼(re_registration 등)을 드롭했는데, 모델 학습에 불필요한 컬럼이라면 스케일링 전에 미리 제거하여 메모리 효율을 높이는 것이 더 좋은 파이프라인 구축 방법이었을 것입니다.

## 배운 점 및 향후 목표
분석 평가지표에 대한 이해: 평가지표가 F1-score일 때, 모델이 반환하는 확률값(predict_proba)을 바탕으로 직접 최적의 임계값을 찾는 코드를 구현해 보면서 분류 문제의 디테일한 성능 튜닝 방법을 체득했습니다.

데이터 품질의 중요성 확인: 극단적인 수치(24시간을 초과하는 투입 시간 등)를 발견하고 EDA를 통해 중앙값으로 통제하는 일련의 과정을 거치며, "Garbage In, Garbage Out"이라는 머신러닝의 기본 원칙을 눈으로 확인할 수 있었습니다.
