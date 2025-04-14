import torch
import openwakeword
import openwakeword.metrics
import matplotlib.pyplot as plt


# Create openWakeWord instance

oww = openwakeword.Model(
    wakeword_models=["/home/openWakeWord/models/hey_thomas20250409.onnx"],
    enable_speex_noise_suppression=True,
    vad_threshold=0.5,
    inference_framework="onnx"
)


# Do a quick test prediction on the test clip to confirm that the behavior is still as expected

scores = oww.predict_clip("/home/openWakeWord/sample_data/my_voice_sample/fixed_male_test1.wav")

print(scores)
plt.figure()
_ = plt.plot([i["hey_thomas_20250409"] for i in scores])
plt.savefig('savefig_default.png')

# # Calculate the false-accept rate per hour from this result

false_accepts = openwakeword.metrics.get_false_positives(
    [i["hey_thomas_20250409"] for i in scores], threshold=0.5
)

print(f"False-accept rate per hour: {false_accepts/1}")