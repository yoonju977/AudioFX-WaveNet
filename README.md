# **Emulation of Guitar Effects Using AI**

**Overview**

This project aims to emulate guitar effects using deep learning, leveraging the WaveNet model to process audio data and generate guitar effects in real time. By training a neural network to learn the transformations applied by various guitar effects, this project seeks to recreate the nuanced sound modifications applied by traditional guitar pedals and audio processors.

# **Methodology and Key Technologies**

# TCN-Based Real-Time Guitar Effect Emulation

This repository contains the implementation and supplementary materials for a Temporal Convolutional Network (TCN)-based model designed to emulate guitar effects in near-real-time. By receiving a clean guitar input signal, the model generates output signals that closely approximate audio produced by professional hardware effect units. This approach integrates advanced preprocessing techniques and a parameter-efficient network architecture, enabling sub-millisecond inference rates and high-fidelity output quality.

## Key Contributions

- **Near-Real-Time Inference**:  
  Achieves average inference times in the range of 1–3 ms, thereby facilitating live performance and recording scenarios with minimal latency.

- **Efficient TCN Architecture**:  
  Employs a TCN optimized for time-series audio analysis. Compared to conventional frameworks such as WaveNet, the TCN model is more parameter-efficient and amenable to parallelization, thereby reducing computational overhead and improving inference speed without compromising audio quality.

- **Enhanced Data Preprocessing**:  
  Incorporates RMS normalization, cross-correlation-based synchronization, STFT-based spectral transformation, and overlapped segmentation. These steps collectively ensure stable input-output alignment and improved frequency-domain representation, contributing to superior signal-to-noise ratio (SNR) and more accurate replication of the target effects.

- **Multi-Modal Loss Integration**:  
  Utilizes a combination of MSE, STFT, and perceptual (Mel-spectrogram) loss functions to account for both time- and frequency-domain characteristics. This strategy ensures that the model does not merely minimize numerical error but also enhances perceptual sound quality.

- **Comprehensive Performance Evaluation**:  
  Evaluates the model using SNR, BFRE (Batch-Level Frequency Response Error), BTAS (Batch-Level Temporal Alignment Score), and PATS (Perceptual A/B Test Score). The proposed solution improves SNR from -3.58 dB in initial configurations to approximately +2.10 dB. In perceptual testing, the model achieves an ~82% preference rating relative to authentic hardware-processed audio signals.

## Repository Structure

- `dataset/`: Placeholder directory for dataset handling scripts and instructions (no raw datasets included).
- `docs/`: Additional documentation, figures, and references detailing architectural design and experimental setups.
- `README.md`: Contains an overview of the project, including high-level methodology, performance indicators, and references.

## Requirements and Setup

**Prerequisites**:
- Python 3.8+
- PyTorch >= 1.7
- torchaudio, numpy, scipy, librosa
- A GPU is recommended for both training and real-time inference.

**Installation**:
```bash
git clone https://github.com/YourUsername/TCN-Guitar-Effect-Emulation.git
cd TCN-Guitar-Effect-Emulation
pip install -r requirements.txt