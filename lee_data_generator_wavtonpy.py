import os
import numpy as np
from tqdm import tqdm
import datasets
from scipy.io.wavfile import write as write_wav


from numpy.lib.format import open_memmap
import openwakeword
from openwakeword.utils import AudioFeatures
import openwakeword
import openwakeword.data


def positive_sample(durations, clip_window_size):
    # Define starting point for each positive clip based on its length, so that each one ends 
    # between 0-200 ms from the end of the total window size chosen for the model.
    # This results in the model being most confident in the prediction right after the
    # end of the wakeword in the audio stream, reducing latency in operation.

    # Get start and end positions for the positive audio in the full window
    sr = 16000
    total_length_seconds = clip_window_size # must be the some window length as that used for the negative examples clip_size랑 같게 맞춰야함.
    total_length = int(sr*total_length_seconds)

    # jitters = (np.random.uniform(0, max_jitter, len(positive_clips))*sr).astype(np.int32)
    # starts = [total_length - (int(np.ceil(i*sr))+j) for i,j in zip(durations, jitters)]
    jitters = [
        int(np.random.uniform(0, max(0, total_length - int(np.ceil(d * sr)))))
        for d in durations
    ]

    starts = [
        max(0, total_length - (int(np.ceil(d * sr)) + j))
        for d, j in zip(durations, jitters)
    ]
    ends = [int(i*sr) + j for i, j in zip(durations, starts)]

    # Create generator to mix the positive audio with background audio
    # foreground_durations 정의 되지 않으면, truncate_clip 수행되지 않음. 
    # 그래서, fg가 원본 size를 가진다. 그러면 fg가 clip_window_size보다(32000) 클 수가 있는데, 그러면 if fg.shape[0] > bg.shape[0]: continue로 인해서 mix가 제외됨.
    # fg가 clip_window_size보다 작으면 상관 없이 그냥 mix됨
    batch_size = 8
    mixing_generator = openwakeword.data.mix_clips_batch(
        foreground_clips = positive_clips,
        background_clips = negative_clips,
        combined_size = total_length,
        batch_size = batch_size,
        snr_low = 5,
        snr_high = 15,
        start_index = starts,
        volume_augmentation=True, # randomly scale the volume of the audio after mixing
    )
    
    N_total = len(positive_clips) # maximum number of rows in mmap file
    n_feature_cols = F.get_embedding_shape(total_length_seconds)
    # output_file = "turn_on_the_office_lights_features.npy"
    output_file = "hey_thomas.npy"
    output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])
    # print(f"N_total: {N_total}, n_feature_cols[0]: {n_feature_cols[0]}, n_feature_cols[1]: {n_feature_cols[1]}")

    # fp is numpy.memmap
    fp = open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_array_shape)
    # print(f"fp: {fp}, fp.shape: {fp.shape}")

    save_count = 0  # wav 저장 카운터
    row_counter = 0
    for batch in tqdm(mixing_generator, total=N_total//batch_size):
        batch, lbls, background = batch[0], batch[1], batch[2]
        
        for i in range(batch.shape[0]):
            if save_count < 10:
                wav_path = f"/home/openWakeWord/sample_data/positive_sample/dubug_positive_{save_count}.wav"
                write_wav(wav_path, 16000, batch[i])
                save_count += 1
        
        
        # ⭐ 1차원 입력일 경우 처리
        if isinstance(batch, list):
            batch = np.stack(batch)
        elif batch.ndim == 1:
            batch = np.expand_dims(batch, axis=0)

        # Compute audio features
        features = F.embed_clips(batch, batch_size=256)

        # Save computed features
        fp[row_counter:row_counter+features.shape[0], :, :] = features
        row_counter += features.shape[0]
        fp.flush()
        
        if row_counter >= N_total:
            break

    # Trip empty rows from the mmapped array
    """
    이 코드는 **메모리 매핑된 numpy 배열(Memory-Mapped Array)**에서 빈 행을 제거하고 새 파일로 저장하는 기능을 수행합니다.
    완전 무음인 데이터 즉 2초라고 치면 32000원소가 모두 0인 데이터를 제외 처리하여 메모리를 절약.
    """
    openwakeword.data.trim_mmap(output_file)




def negative_sample(audio_dataset, clip_window_size):
    # Get audio embeddings (features) for negative clips and save to .npy file
    # Process files by batch and save to Numpy memory mapped file so that
    # an array larger than the available system memory can be created

    batch_size = 64 # number of files to load, compute features, and write to mmap at a time
    clip_size = clip_window_size  # the desired window size (in seconds) for the trained openWakeWord model 모델에게 학습시킬 입력 단위는 "n초 길이의 오디오로 설정하는 역할"
    N_total = int(sum(negative_durations)//clip_size) # maximum number of rows in mmap file
    n_feature_cols = F.get_embedding_shape(clip_size)
    save_count = 0
    
    output_file = "negative_features.npy"
    output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])
    fp = open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_array_shape)

    row_counter = 0
    for i in tqdm(np.arange(0, audio_dataset.num_rows, batch_size)):
        # Load data in batches and shape into rectangular array
        # N개의 duration이 다른 음성 샘플이 존재하면, 그걸 16000*clip_size 길이에 맞춰서 배치 데이터 생성
        wav_data = [(j["array"]*32767).astype(np.int16) for j in audio_dataset[i:i+batch_size]["audio"]]
        wav_data = openwakeword.data.stack_clips(wav_data, clip_size=16000*clip_size).astype(np.int16)
        
        for j in range(wav_data.shape[0]):
            if save_count < 10:
                wav_path = f"/home/openWakeWord/sample_data/negative_sample/dubug_negative_{save_count}.wav"
                write_wav(wav_path, 16000, wav_data[j])
                save_count += 1
        
        # Compute features (increase ncpu argument for faster processing)
        features = F.embed_clips(x=wav_data, batch_size=1024, ncpu=8)
                
        # Save computed features to mmap array file (stopping once the desired size is reached)
        # fp.flush()의 역할 : fp는 numpy의 open_memmap()으로 만든 memory-mapped 파일 객체, 데이터를 실제 .npy 파일에 조금씩 직접 저장하는 방식으로 작동, 데이터는 먼저 메모리에 저장되고, flush()를 호출해야 실제로 디스크에 반영
        # 훈련 도중 오류/중단되더라도 저장된 데이터가 유실되지 않도록 하기 위함
        # 메모리에 너무 많은 데이터를 쌓는 걸 방지하기 위함
        if row_counter + features.shape[0] > N_total:
            fp[row_counter:min(row_counter+features.shape[0], N_total), :, :] = features[0:N_total - row_counter, :, :]
            fp.flush()
            break
        else:
            fp[row_counter:row_counter+features.shape[0], :, :] = features
            row_counter += features.shape[0]
            fp.flush()
        print(f"features.shape: {features.shape}")
        print(f"row_counter: {row_counter}")
            
    # Trip empty rows from the mmapped array
    openwakeword.data.trim_mmap(output_file)


if __name__ == "__main__":
    
    # dataset loading
    negative_clips, negative_durations = openwakeword.data.filter_audio_paths([
                                                                            #    "/home/data/openwakeword/common_voice_Corpus_21.0/cv-corpus-21.0-2025-03-14/en/60sec_below_negative_clips",
                                                                            #    "/home/data/openwakeword/background_sound/audioset_16k",
                                                                            #    "/home/data/openwakeword/background_sound/custom_data",
                                                                            #    "/home/data/openwakeword/background_sound/fma",
                                                                            "/home/data/openwakeword/ai_hub_free_conversation_voice/일반남여_자유대화_M_1578412985_32_수도권_실내",
                                                                            "/home/data/openwakeword/ai_hub_free_conversation_voice/일반남여_자유대화_M_1571674136_39_경상_실내",
                                                                            "/home/data/openwakeword/ai_hub_free_conversation_voice/일반남여_일반통합12_M_1577013323_36_수도권_실내",
                                                                            "/home/data/openwakeword/ai_hub_free_conversation_voice/일반남여_일반통합12_M_1573837987_44_경상_실내"
                                                                            
                                                                               ], # data path 
                                                                              min_length_secs = 0.5, # minimum clip length in seconds
                                                                              max_length_secs = 60*2, # maximum clip length in seconds
                                                                              duration_method = "header" # use the file header to calculate duration)
    )
    positive_clips, positive_durations = openwakeword.data.filter_audio_paths([
                                                                            #    "/home/data/openwakeword/openwakeword_positive_sample/piper_VITS_n_mix5_ratio0.5",
                                                                            #    "/home/data/openwakeword/openwakeword_positive_sample/VITS_n_mix5_ratio0.8",
                                                                            #    "/home/data/openwakeword/openwakeword_positive_sample/VITS_n_mix5_ratio0.2",
                                                                            #    "/home/data/openwakeword/openwakeword_positive_sample/VITS_n_mix4_ratio0.5",
                                                                               "/home/data/openwakeword/openwakeword_positive_sample/VITS_n_mix2_ratio0.2"
                                                                               ], # data path 
                                                                              min_length_secs = 0.5, # minimum clip length in seconds
                                                                              max_length_secs = 30*1, # maximum clip length in seconds
                                                                              duration_method = "header" # use the file header to calculate duration)
    )


    audio_dataset = datasets.Dataset.from_dict({"audio": negative_clips})
    audio_dataset = audio_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

    # 오디오 데이터를 특징 벡터로 변환하는 파이프라인의 핵심 구성 요소
    F = AudioFeatures() # -> 이거 할때 64000이 저장되네?
    
    audio_window_size = 2
    
    negative_sample(audio_dataset, audio_window_size)
    positive_sample(positive_durations, audio_window_size)
    
    
    