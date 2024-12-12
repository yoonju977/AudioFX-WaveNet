# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import numpy as np
from tqdm import tqdm
from collections import deque
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write
os.environ["SD_ENABLE_ASIO"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
recorded_output = []
#모델을 로드해서 사용하기때문에 같은 모델구조 정의
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.final_layer = nn.Conv1d(num_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        out = self.network(x)
        out = self.final_layer(out)
        return out

def calculate_receptive_field(num_channels, kernel_size):
    receptive_field = 1
    for i in range(len(num_channels)):
        dilation = 2 ** i
        receptive_field += (kernel_size - 1) * dilation
    return receptive_field
def lowpass_filter(data, cutoff=9000, fs=44100, order=5):#소리가 많이 달라져 사용하지 않음
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


class RealTimeProcessor:
    def __init__(self, model, segment_length, receptive_field, device, overlap=0.5, filter_cutoff=9000, sample_rate=44100):
        self.model = model.to(device)
        self.segment_length = segment_length
        self.receptive_field = receptive_field
        self.device = device
        self.overlap = overlap
        self.filter_cutoff = filter_cutoff  
        self.sample_rate = sample_rate     
        self.buffer = deque(maxlen=receptive_field + segment_length)


    def process(self, audio_chunk):
        if len(self.buffer) < self.receptive_field + self.segment_length:
            padding_needed = self.receptive_field + self.segment_length - len(self.buffer)
            self.buffer.extend([0] * padding_needed)

        self.buffer.extend(audio_chunk.tolist())
        inputs = list(self.buffer)
        signal = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(signal).cpu().squeeze().numpy()
        # 로우패스 필터 적용
        filtered_output = lowpass_filter(output, cutoff=self.filter_cutoff, fs=self.sample_rate)
        return filtered_output


# 실시간 오디오 처리 콜백 함수
def audio_callback(indata, outdata, frames, time, status, processor):
    global recorded_output
    if status:
        print(f"Stream status: {status}")

    mono_input = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
    output = processor.process(mono_input)

    if len(output) < frames:
        output = np.pad(output, (0, frames - len(output)), mode='constant')
    else:
        output = output[:frames]

    stereo_output = np.column_stack((output, output))
    outdata[:] = stereo_output
    recorded_output.extend(output.tolist())

# 메인 함수
def main():
    output_file_path = "C:\\Users\\LSH\\Desktop\\recorded_output.wav"
    global recorded_output
    model_path = "C:\\Users\\LSH\\Desktop\\v2_best.pth"
    sample_rate = 44100
    block_size = 512
    num_channels = [32, 32, 32, 32]
    kernel_size = 3
    dropout = 0.2

    receptive_field = calculate_receptive_field(num_channels, kernel_size)
    print(f"Receptive field: {receptive_field}")

    model = TemporalConvNet(num_inputs=1, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
    
 
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) 
    model.eval()

    processor = RealTimeProcessor(model, segment_length=block_size, receptive_field=receptive_field, device=device)

    with sd.Stream(
        device=(16,16),
        samplerate=sample_rate,
        blocksize=block_size,
        channels=2,
        dtype="float32",
        latency="low",
        callback=lambda indata, outdata, frames, time, status: audio_callback(indata, outdata, frames, time, status, processor),
    ):
        print("Real-time audio processing started. Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Stopping... Saving recorded output to file.")
            output_array = np.array(recorded_output, dtype=np.float32)
            write(output_file_path, sample_rate, output_array)
            print("File saved as 'recorded_output.wav'.")

if __name__ == "__main__":
    main()
