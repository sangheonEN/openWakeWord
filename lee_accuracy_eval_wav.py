import torch
import openwakeword
import openwakeword.metrics
import matplotlib.pyplot as plt

"""
이 코드는 내가 만든 커스텀 모델에 대해서 accuracy 평가를 수행한다.

평가에 사용될 wav 파일을 가지고 모델에 입력한 뒤에 모델의 prediction 결과를 시각화한다.

"""



# Create openWakeWord instance

oww = openwakeword.Model(
    wakeword_models=["/home/openWakeWord/my_custom_model/hey_thomas.onnx"],
    enable_speex_noise_suppression=True,
    vad_threshold=0.5,
    inference_framework="onnx"
)


# Do a quick test prediction on the test clip to confirm that the behavior is still as expected

scores = oww.predict_clip("/home/openWakeWord/sample_data/test_sample/positive_sample/lee_hey_thomas_mix.wav")

print(scores)
plt.figure()
_ = plt.plot([i["hey_thomas"] for i in scores])
plt.savefig('savefig_default.png')

# # Calculate the false-accept rate per hour from this result

false_accepts = openwakeword.metrics.get_false_positives(
    [i["hey_thomas"] for i in scores], threshold=0.5
)

print(f"False-accept rate per hour: {false_accepts/1}")