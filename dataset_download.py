# Download room impulse responses collected by MIT
# https://mcdermottlab.mit.edu/Reverb/IR_Survey.html
"""
mit_rirs 데이터셋 다운로드 코드

import os
import datasets
import scipy
import tqdm
import numpy as np

output_dir = "./mit_rirs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
rir_dataset = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)

# Save clips to 16-bit PCM wav files
for row in tqdm.tqdm(rir_dataset):
    name = row['audio']['path'].split('/')[-1]
    scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
    
"""

"""
#Free Music Archive dataset 다운로드 코드

# Free Music Archive dataset (https://github.com/mdeff/fma)

# datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True) 여기서 streaming=True 옵션을 안주고 다운로드 해야해서 로컬에 데이터 다운로드함.
# ~/.cache/huggingface/datasets/rudraml__fma 경로에 데이터 저장되어 있음.


import os
import datasets
import scipy
import tqdm
import numpy as np

dc = datasets.DownloadConfig(
    max_retries=10,          # 실패 시 재시도
    resume_download=True,    # 부분 다운로드 이어받기
    # timeout=...            # 일부 버전은 timeout 지원(노트 아래)
)

# Free Music Archive dataset (https://github.com/mdeff/fma)
output_dir = "./fma"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", download_config=dc, streaming=True)
fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))

n_hours = 1  # use only 1 hour of clips for this example notebook, recommend increasing for full-scale training
for i in tqdm.tqdm(range(n_hours*3600//30)):  # this works because the FMA dataset is all 30 second clips
    row = next(fma_dataset)
    name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
    scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
    i += 1
    if i == n_hours*3600//30:
        break
"""