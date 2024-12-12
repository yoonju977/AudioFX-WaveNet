import logging
import time  # 시간 측정을 위해 추가

import numpy as np
import torch
import torchaudio
from scipy.signal import butter, lfilter
from torch.utils.data import Dataset

# 로깅 설정
logging.basicConfig(
    filename="audio_processing.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting audio processing script.")

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# STFT 기반 전처리
def compute_stft(audio, n_fft=512, hop_length=128, window=None):
    logging.debug("Computing STFT.")
    if window is None or window.device != audio.device:
        window = torch.hann_window(n_fft, device=audio.device)
    stft = torch.stft(
        audio.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    return magnitude, phase

# RMS 정규화
def rms_normalize(audio):
    start_time = time.time()
    logging.debug("Performing RMS normalization.")
    rms = torch.sqrt(torch.mean(audio**2))
    normalized_audio = audio / (rms + 1e-8)  # 분모에 작은 값을 더해 안정성 확보
    logging.debug(f"RMS normalization completed in {time.time() - start_time:.2f}s.")
    return normalized_audio


# RMS 기반 볼륨 일치
def match_rms(clean_audio, effect_audio):
    logging.debug("Matching RMS volumes.")
    clean_rms = torch.sqrt(torch.mean(clean_audio**2))
    effect_rms = torch.sqrt(torch.mean(effect_audio**2))
    scaling_factor = clean_rms / (effect_rms + 1e-8)
    return effect_audio * scaling_factor


# 오버랩 적용된 윈도우 세그먼트
def segment_with_overlap(audio, target_sr, segment_length, overlap=0.5):
    logging.debug("Creating segments with overlap.")
    segment_samples = int(segment_length * target_sr)
    step = int(segment_samples * (1 - overlap))  # 오버랩 비율 반영
    segments = [
        audio[:, i : min(i + segment_samples, audio.size(1))]
        for i in range(0, audio.size(1), step)
    ]
    logging.debug(f"Generated {len(segments)} segments.")
    return segments


# 동기화 (Cross-Correlation 기반)
def synchronize_signals(clean_audio, effect_audio):
    logging.debug("Synchronizing signals.")
    # 시간 길이가 같으면 동기화 생략
    if clean_audio.size(1) == effect_audio.size(1):
        logging.debug("Signals have the same length. Skipping synchronization.")
        return clean_audio, effect_audio

    # Cross-Correlation 수행
    logging.debug("Performing Cross-Correlation for synchronization.")
    clean_np = clean_audio.squeeze(0).cpu().numpy()
    effect_np = effect_audio.squeeze(0).cpu().numpy()
    correlation = np.correlate(clean_np, effect_np, mode="full")
    lag = np.argmax(correlation) - len(clean_np) + 1

    # Effect 신호를 lag만큼 이동하여 동기화
    if lag > 0:
        effect_audio = effect_audio[:, lag:]
    elif lag < 0:
        clean_audio = clean_audio[:, -lag:]

    # 최소 길이로 맞춤
    min_length = min(clean_audio.size(1), effect_audio.size(1))
    clean_audio = clean_audio[:, :min_length]
    effect_audio = effect_audio[:, :min_length]

    return clean_audio, effect_audio

# 오디오 전처리 함수
def preprocess_audio(audio, sr, target_sr):
    logging.debug("Preprocessing audio.")
    start_time = time.time()
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    normalized_audio = rms_normalize(audio)
    logging.debug(f"Audio preprocessing completed in {time.time() - start_time:.2f}s.")
    return normalized_audio.to(device)


# 데이터셋 클래스
class AudioDataset(Dataset):
    def __init__(
        self, clean_files, effect_files, target_sr=16000, segment_length=2, overlap=0.5
    ):
        logging.info("Initializing AudioDataset.")
        self.clean_segments = []
        self.effect_segments = []
        self.stft_window = torch.hann_window(512, device=device)  # STFT 윈도우 캐싱

        # 모든 파일의 길이가 같은지 확인
        self.skip_synchronization = True
        for idx, (clean, effect) in enumerate(zip(clean_files, effect_files)):
            logging.info(f"Processing pair {idx+1}: {clean}, {effect}")
            try:
                clean_audio, sr = torchaudio.load(clean)
                effect_audio, _ = torchaudio.load(effect)

                clean_audio = preprocess_audio(clean_audio, sr, target_sr)
                effect_audio = preprocess_audio(effect_audio, sr, target_sr)

                # 길이 확인: 하나라도 길이가 다르면 동기화 필요
                if clean_audio.size(1) != effect_audio.size(1):
                    self.skip_synchronization = False
                    break  # 동기화 필요가 확인되면 루프 종료
            except Exception as e:
                logging.error(f"Error processing files: {clean}, {effect}. Error: {e}")
                continue

        logging.info(
            f"All signals have equal length: {self.skip_synchronization}. Synchronization {'skipped' if self.skip_synchronization else 'required'}."
        )

        for idx, (clean, effect) in enumerate(zip(clean_files, effect_files)):
            try:
                clean_audio, sr = torchaudio.load(clean)
                effect_audio, _ = torchaudio.load(effect)

                clean_audio = preprocess_audio(clean_audio, sr, target_sr)
                effect_audio = preprocess_audio(effect_audio, sr, target_sr)

                # 동기화 단계
                if not self.skip_synchronization:
                    clean_audio, effect_audio = synchronize_signals(
                        clean_audio, effect_audio
                    )

                # RMS 볼륨 일치
                effect_audio = match_rms(clean_audio, effect_audio)

                # 윈도우 세그먼트 생성 (오버랩 포함)
                clean_segs = segment_with_overlap(
                    clean_audio, target_sr, segment_length, overlap
                )
                effect_segs = segment_with_overlap(
                    effect_audio, target_sr, segment_length, overlap
                )

                if len(clean_segs) != len(effect_segs):
                    raise ValueError(
                        "Mismatch in segment lengths: clean_segs and effect_segs."
                    )

                self.clean_segments.extend(clean_segs)
                self.effect_segments.extend(effect_segs)
            except Exception as e:
                logging.error(f"Error processing files: {clean}, {effect}. Error: {e}")
                continue

    def __len__(self):
        return len(self.clean_segments)

    def __getitem__(self, idx):
        clean_segment = self.clean_segments[idx]
        effect_segment = self.effect_segments[idx]

        clean_magnitude, clean_phase = compute_stft(
            clean_segment, n_fft=512, hop_length=128, window=self.stft_window
        )
        effect_magnitude, effect_phase = compute_stft(
            effect_segment, n_fft=512, hop_length=128, window=self.stft_window
        )

        return clean_magnitude, clean_phase, effect_magnitude, effect_phase


# 테스트를 위한 샘플 데이터 로드
clean_files = [
    "./dataset/fenderneckScNoEffect.wav",
    "./dataset/ibanezStratCleanB+NnoEffect.wav",
    "./dataset/ibanezStratCleanNeckNoEffect.wav",
    "./dataset/ibanzeHuNoEffect.wav",
    "./dataset/scChordsNoEffect.wav",
    "./dataset/noEffect.wav",
    "./dataset/noeffect hex-pickup.wav",
]

effect_files = [
    "./dataset/fenderneckScEffect.wav",
    "./dataset/ibanezStratCleanB+NEffect.wav",
    "./dataset/ibanezStratCleanNeckEffect.wav",
    "./dataset/ibanzeHuEffect.wav",
    "./dataset/scChordsEffect.wav",
    "./dataset/effect.wav",
    "./dataset/effect hex-pickup.wav",
]

test_clean_files = ["./testset/dataset3NoEffect.wav"]
test_effect_files = ["./testset/dataset3Effect.wav"]

# 데이터셋 생성 및 확인
dataset = AudioDataset(
    clean_files, effect_files, target_sr=16000, segment_length=2, overlap=0.5
)
print(f"Dataset size: {len(dataset)}")

# 데이터 샘플 확인
sample = dataset[0]
print(f"Clean Magnitude Shape: {sample[0].shape}, Clean Phase Shape: {sample[1].shape}")
print(
    f"Effect Magnitude Shape: {sample[2].shape}, Effect Phase Shape: {sample[3].shape}"
)

# 학습 데이터 전처리 후 저장
dataset = AudioDataset(
    clean_files, effect_files, target_sr=16000, segment_length=2, overlap=0.5
)
torch.save(dataset, "./process_set/preprocessed_train_data.pt")

# 테스트 데이터 전처리 후 저장
test_dataset = AudioDataset(
    test_clean_files, test_effect_files, target_sr=16000, segment_length=2, overlap=0.5
)
torch.save(test_dataset, "./process_testset/preprocessed_test_data.pt")


# 검증
try:
    # 학습 데이터 로드
    loaded_train_dataset = torch.load("./process_set/preprocessed_train_data.pt")
    print(f"Loaded Train Dataset Size: {len(loaded_train_dataset)}")

    # 데이터 타입 확인
    print(f"First Sample Type: {type(loaded_train_dataset[0])}")
    print(
        f"First Sample Shapes: Clean Magnitude: {loaded_train_dataset[0][0].shape}, Clean Phase: {loaded_train_dataset[0][1].shape}, Effect Magnitude: {loaded_train_dataset[0][2].shape}, Effect Phase: {loaded_train_dataset[0][3].shape}"
    )

except FileNotFoundError as e:
    print(f"Error: {e}. Check if the path is correct and the file exists.")
except Exception as e:
    print(f"Unexpected error: {e}")
