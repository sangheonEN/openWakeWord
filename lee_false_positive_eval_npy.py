import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
이 코드는 False Positive 평가를 위해서 target_phrase가 포함되지 않은 오디오 데이터에 대해서 검증한다.

결과는 negative_features_false_positive_eval.png로 저장됨. 아무 감지가 되지 않아야 정상이다!

내가 lee_data_generator_wavtonpy로 생성하는 데이터는 하나의 window_size초 오디오로부터 16개의 임베딩 벡터 (96차원)가 생성 (n, 16, 96)
즉, 1 데이터당 window_size초 짜리임!

"""

# 1. Load features from file (shape: [N, 16, 96] or [N, 96])
# npy_path = "/home/openWakeWord/negative_features.npy"
npy_path = "/home/data/openwakeword/npy_data/negative_sample/0001.npy"
features = np.load(npy_path)
use_sliding_window = True if features.ndim == 2 else False

# 2. Load ONNX model
onnx_model_path = "/home/openWakeWord/my_custom_model/hey_thomas.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name


if use_sliding_window:
        
    window_size = 16
    scores = []
    for i in tqdm(range(0, features.shape[0] - window_size)):
        window = features[i:i + window_size].astype(np.float32).reshape(1, window_size, 96)
        output = session.run(None, {input_name: window})
        scores.append(float(output[0][0][0]))  # output shape: (1, 1)
        
else:    
    # 3. 예측 수행 (슬라이딩 필요 없음)
    scores = []
    for i in tqdm(range(features.shape[0])):
        window = features[i].astype(np.float32).reshape(1, 16, 96)  # 이미 16x96
        output = session.run(None, {input_name: window})
        scores.append(float(output[0][0][0]))  # output shape: (1, 1)

# 4. 결과 시각화
plt.figure(figsize=(12, 4))
plt.plot(scores, label="False Positive Confidence"  )
plt.axhline(0.5, color='red', linestyle='--', label="Threshold = 0.5")
plt.xlabel("Window Index")
plt.ylabel("Confidence")
plt.title("False Positive Scores on Validation Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("negative_features_false_positive_eval.png")
plt.show()