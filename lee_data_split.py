import os
import shutil
import soundfile as sf
from tqdm import tqdm
import json

# 원본 디렉토리
source_dir = "/home/data/openwakeword/common_voice_Corpus_21.0/cv-corpus-21.0-2025-03-14/en/wav_clips"
# 필터링된 파일을 저장할 디렉토리
target_dir = "/home/data/openwakeword/common_voice_Corpus_21.0/cv-corpus-21.0-2025-03-14/en/60sec_below_negative_clips"
os.makedirs(target_dir, exist_ok=True)  # ✅ 디렉토리 없으면 자동 생성

# 필터 기준
min_duration = 0.5  # seconds
max_duration = 60.0
max_files = 120_000

copied_files = []

for fname in tqdm(os.listdir(source_dir)):
    if not fname.endswith(".wav"):
        continue
    fpath = os.path.join(source_dir, fname)
    
    try:
        info = sf.info(fpath)
        duration = info.frames / info.samplerate
        if min_duration <= duration <= max_duration:
            shutil.copy(fpath, os.path.join(target_dir, fname))
            copied_files.append({
                "filename": fname,
                "duration": round(duration, 3)
            })
            if len(copied_files) >= max_files:
                break
    except Exception as e:
        print(f"Error reading {fpath}: {e}")

# 요약 정보 + 파일 리스트 저장
summary = {
    "source_dir": source_dir,
    "target_dir": target_dir,
    "total_copied": len(copied_files),
    "min_duration": min_duration,
    "max_duration": max_duration,
    "files": copied_files  # 만약 너무 많으면 일부만 잘라서 저장할 수도 있음
}

json_path = os.path.join(target_dir, "filtered_summary.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"✅ 총 {len(copied_files)}개의 파일이 {target_dir}에 복사되었고, 요약 정보는 {json_path}에 저장되었습니다.")