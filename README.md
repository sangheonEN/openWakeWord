# Updates

# SRC 설명

1. lee_custom_train.py : automatic_model_training.ipynb 기반의 학습 방법을 벤치마킹하여 구축 중
    - /home/openWakeWord/config/training_config.yml 에서 train config 설정
    - /home/openWakeWord/my_custom_model의 경로에 model_name: "hey_thomas_20250409_max_negative_weight250" 폴더를 생성하고 거기에 npy 데이터를 넣어줘야한다.
    """예시 
    feature_data_files:
        negative_features_train: "/home/openWakeWord/my_custom_model/hey_thomas_20250409/negative_features_train.npy"
        negative_features_test: "/home/openWakeWord/my_custom_model/hey_thomas_20250409/negative_features_test.npy"
        positive_features_train: "/home/openWakeWord/my_custom_model/hey_thomas_20250409/positive_features_train.npy"
        positive_features_test: "/home/openWakeWord/my_custom_model/hey_thomas_20250409/positive_features_test.npy"
    """
2. lee_data_generator_wavtonpy.py : training_models.ipynb에서 positive, negative wav 파일을 npy 형식으로 변환하는 코드. (positive sample 변환 시 mix_clip 적용)
3. lee_data_split.py : wav파일을 탐색해서 특정 시간 내에 음성 데이터를 추출하여 저장하는 코드.
4. lee_evaluator.py : 학습된 wakeword onnx 모델 파일을 loading 해서 wav 파일을 입력으로 하여 추론 정확도를 그래프 분석 및 평가 코드.

**2025/04/10**

현재까지, lee_custom_train.py와 training_models.ipynb 두 학습 방법에 대해서 학습을 진행했음.

동일한 데이터를 활용해서 진행했지만, training_models.ipynb로 학습한 모델은 어느정도 성능이 나오긴 하는데, lee_custom_train.py 방법으로 학습한 모델은 아예 성능이 안나옴.

두 학습 방법의 차이점을 파악해서 training_models.ipynb 방법으로 학습하는 것을 개선할 수 있는 방법이 있다면, 개선해야할듯.

일단, RealTimeSTT를 활용해서 Wake word 모델을 테스트해봐야함.