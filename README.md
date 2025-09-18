# Updates

# SRC 설명

1. lee_custom_train.py : automatic_model_training.ipynb 기반의 학습 방법을 벤치마킹하여 구축 중
    - /home/openWakeWord/config/training_config.yml 에서 train config 설정
    - /home/openWakeWord/my_custom_model의 경로에 model_name: "hey_thomas_20250409_max_negative_weight250" 폴더를 생성하고 거기에 npy 데이터를 넣어줘야한다.
    """예시 
    feature_data_files: 깃헙에는 용량 문제로 인해 별도로 빼둠. /mnt/d/data/openwakeword_train_result/20250409_openwakeword.zip
        negative_features_train: "/home/openWakeWord/my_custom_model/hey_thomas_20250409/negative_features_train.npy"
        negative_features_test: "/home/openWakeWord/my_custom_model/hey_thomas_20250409/negative_features_test.npy"
        positive_features_train: "/home/openWakeWord/my_custom_model/hey_thomas_20250409/positive_features_train.npy"
        positive_features_test: "/home/openWakeWord/my_custom_model/hey_thomas_20250409/positive_features_test.npy"
    """
2. lee_data_generator_wavtonpy.py : training_models.ipynb에서 positive, negative wav 파일을 npy 형식으로 변환하는 코드. (positive sample 변환 시 mix_clip 적용)
3. lee_data_split.py : wav파일을 탐색해서 특정 시간 내에 음성 데이터를 추출하여 저장하는 코드.
4. lee_accuracy_eval_wav.py : 학습된 wakeword onnx 모델 파일을 loading 해서 wav 파일을 입력으로 하여 추론 정확도를 그래프 분석 및 평가 코드.
4. lee_false_positive_eval_npy.py : 학습된 wakeword onnx 모델 파일을 loading 해서 npy 파일을 입력으로 하여 추론 정확도를 그래프 분석 및 평가 코드.

**2025/04/10**

현재까지, lee_custom_train.py와 training_models.ipynb 두 학습 방법에 대해서 학습을 진행했음.

동일한 데이터를 활용해서 진행했지만, training_models.ipynb로 학습한 모델은 어느정도 성능이 나오긴 하는데, lee_custom_train.py 방법으로 학습한 모델은 아예 성능이 안나옴.

두 학습 방법의 차이점을 파악해서 training_models.ipynb 방법으로 학습하는 것을 개선할 수 있는 방법이 있다면, 개선해야할듯.

일단, RealTimeSTT를 활용해서 Wake word 모델을 테스트해봐야함.

모델을 분석한 결과는 notion에 정리되어 있고, 현재까지 False Positive 문제로 인해서 활용하기 어렵다.

한국어 negative sample에서는 조금 덜하지만, False Positive가 발생한다.

영어 negative sample에서는 거의 모든 샘플에서 False Positive가 발생했다. (근데 이건 오픈 소스에서 제공하는 npy 데이터를 활용해서 그런거 아닌가 싶기도 하고..)


**2025/09/19**

A6000 서버에 d/data/openwakeword_train_result 경로에 일자별로 학습 결과 데이터 저장.

우선 소스 별로 기능 정리를 좀 해야겠다.

1. 로컬 디스크 드라이버에 데이터셋 저장 폴더 경로를 참조해서 학습을 바로 진행하는 코드 : /home/openWakeWord/lee_train_code/automatic_model_training_simple_leesangheon.py

2. false positive evaluation 코드 : lee_false_positive_eval_npy.py

3. wav파일을 npy파일로 변환하는 코드 : lee_data_generator_wavtonpy.py

4. prediction 기준 accuracy 측정 코드 : lee_accuracy_eval_wav.py