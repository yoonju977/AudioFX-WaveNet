import logging
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader

from audio_processing import AudioDataset

# 로깅 설정
logging.basicConfig(
    filename="tcn_training.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("TCN training script started.")

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_input_for_stft(input_tensor, n_fft=512, device="cuda"):
    """패딩 후 텐서를 디바이스로 이동"""
    seq_length = input_tensor.size(-1)
    min_length = n_fft // 2 + 1  # STFT가 요구하는 최소 길이

    if seq_length < min_length:
        pad_size = min_length - seq_length
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, pad_size), mode="constant", value=0
        )
        print(f"Padded input from {seq_length} to {min_length}.")

    return input_tensor.to(device)  # 디바이스 이동


# STFT Loss 정의
stft_window = None  # 전역 윈도우 변수 추가


def stft_loss(output, target, n_fft=256, hop_length=128, device="cuda"):
    global stft_window  # 전역 윈도우 변수 사용
    logging.debug("Computing STFT Loss.")

    # 입력 데이터 패딩 후 디바이스 이동
    output = pad_input_for_stft(output, n_fft=n_fft, device=device)
    target = pad_input_for_stft(target, n_fft=n_fft, device=device)

    # STFT 계산에 사용할 윈도우 생성
    if stft_window is None or stft_window.device != device:
        stft_window = torch.hann_window(n_fft, device=device)

    print(
        f"Output device: {output.device}, Target device: {target.device}, Window device: {stft_window.device}"
    )

    # STFT를 각 배치/채널에 대해 개별적으로 계산
    batch_size, channels, time_steps = output.shape
    output_stft_list = []
    target_stft_list = []

    for b in range(batch_size):
        for c in range(channels):
            output_stft = torch.stft(
                output[b, c, :],
                n_fft=n_fft,
                hop_length=hop_length,
                window=stft_window,
                return_complex=True,
            )
            target_stft = torch.stft(
                target[b, c, :],
                n_fft=n_fft,
                hop_length=hop_length,
                window=stft_window,
                return_complex=True,
            )
            output_stft_list.append(output_stft)
            target_stft_list.append(target_stft)

    # 텐서를 다시 합치기
    output_stft_tensor = torch.stack(output_stft_list).view(batch_size, channels, -1)
    target_stft_tensor = torch.stack(target_stft_list).view(batch_size, channels, -1)

    # STFT 손실 계산
    return torch.mean(torch.abs(output_stft_tensor - target_stft_tensor))


# Perceptual Loss 정의
def perceptual_loss(output, target):
    logging.debug("Computing Perceptual Loss.")
    mel_spec = torchaudio.transforms.MelSpectrogram().to(device)
    output_mel = mel_spec(output.to(device))
    target_mel = mel_spec(target.to(device))
    return torch.mean((output_mel - target_mel) ** 2)


# 데이터셋 로드 및 input_size 추출
train_dataset = torch.load("./process_set/preprocessed_train_data.pt")
input_size = train_dataset[0][0].shape[1]
print(f"Extracted input_size: {input_size}")


class TemporalBlock(nn.Module):
    """TCN의 기본 블록"""

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        input_size,
        dropout=0.2,
    ):
        super(TemporalBlock, self).__init__()
        # 패딩 크기를 입력 데이터의 길이를 초과하지 않도록 제한
        padding = min((kernel_size - 1) * dilation // 2, input_size // 2, 125)
        print(
            f"Calculated padding: {padding}, Kernel size: {kernel_size}, Dilation: {dilation}, Input size: {input_size}"
        )

        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

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
    """TCN 모델"""

    def __init__(
        self, num_inputs, input_size, num_channels, kernel_size=2, dropout=0.2
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i  # 팽창률
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    input_size=input_size,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)
        self.final_layer = nn.Conv1d(num_channels[-1], 257, kernel_size=1)

    def forward(self, x):
        out = self.network(x)
        out = self.final_layer(out)
        return out


# ------------------Pruning 및 Fine-tuning 추가----------------------


def prune_and_fine_tune(
    model, train_loader, val_loader, pruning_amount=0.2, fine_tune_epochs=5, lr=1e-5
):
    logging.info(f"Applying Pruning with amount: {pruning_amount}")

    # Pruning 적용
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=pruning_amount)

    # Pruning 후 마스크 제거
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.remove(module, "weight")

    logging.info("Pruning applied. Starting Fine-tuning...")

    # Fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(fine_tune_epochs):
        epoch_loss = 0
        for clean, effect in train_loader:
            clean, effect = clean.to(device), effect.to(device)
            optimizer.zero_grad()

            output = model(clean)
            loss = (
                criterion(output, effect)
                + stft_loss(output, effect, device=device)
                + perceptual_loss(output, effect)
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        logging.info(
            f"Fine-tuning Epoch {epoch+1}/{fine_tune_epochs}, Loss: {avg_loss:.6f}"
        )

    # Fine-tuned 모델 저장
    torch.save(model.state_dict(), "./models/fine_tuned_model.pth")
    logging.info("Fine-tuned model saved.")


# ------------------ Pruning 및 Fine-tuning 종료 ------------------


# 학습 및 검증 루프
def train_and_validate(
    model, train_loader, val_loader, epochs, lr, patience, experiment_name, tcn_params
):
    logging.info(f"Starting training for experiment: {experiment_name}")
    start_time = time.time()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for clean_magnitude, _, effect_magnitude, _ in train_loader:
            print(f"Shape before permute: {clean_magnitude.shape}")
            clean_magnitude = clean_magnitude.permute(0, 1, 2).to(device)
            print(f"Shape after permute: {clean_magnitude.shape}")
            effect_magnitude = effect_magnitude.permute(0, 1, 2).to(device)

            optimizer.zero_grad()

            output = model(clean_magnitude)
            loss = (
                criterion(output, effect_magnitude)
                + stft_loss(output, effect_magnitude, device=device)
                + perceptual_loss(output, effect_magnitude)
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")

        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for clean_magnitude, _, effect_magnitude, _ in val_loader:
                clean_magnitude = clean_magnitude.permute(0, 1, 2).to(device)
                effect_magnitude = effect_magnitude.permute(0, 1, 2).to(device)

                output = model(clean_magnitude)
                val_loss += (
                    criterion(output, effect_magnitude)
                    + stft_loss(output, effect_magnitude, device=device)
                    + perceptual_loss(output, effect_magnitude)
                ).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.6f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"./models/{experiment_name}_best_model.pth")
            logging.info(
                f"Model saved at epoch {epoch+1} with Val Loss: {avg_val_loss:.6f}"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered.")
                break

        scheduler.step()

    total_time = time.time() - start_time
    logging.info(
        f"Experiment '{experiment_name}' completed in {total_time:.2f} seconds."
    )
    save_results(train_losses, val_losses, total_time, experiment_name, tcn_params)

    return train_losses, val_losses


# 결과 저장 함수
def save_results(train_losses, val_losses, total_time, experiment_name, tcn_params):
    logging.info(f"Saving results for experiment: {experiment_name}")
    # 그래프 저장
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Curve for {experiment_name}")
    plt.savefig(f"./results/{experiment_name}_loss_curve.png")
    logging.info(f"Loss curve saved for experiment: {experiment_name}")

    # 텍스트 파일 저장
    with open(f"./results/{experiment_name}_results.txt", "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"TCN Parameters: {tcn_params}\n")
        f.write(f"Total Time: {total_time:.2f} seconds\n")
        f.write(f"Final Train Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.6f}\n")


# main함수
def main():
    logging.info("Loading datasets.")
    # 데이터 로드
    try:
        train_dataset = torch.load("./process_set/preprocessed_train_data.pt")
        test_dataset = torch.load("./process_testset/preprocessed_test_data.pt")
        logging.info("Datasets loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        return

    # 데이터셋 로드 및 input_size 추출
    input_size = train_dataset[0][0].shape[1]

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 실험 설정
    tcn_experiments = [
        {"num_channels": [32, 64, 128], "kernel_size": 3, "dropout": 0.2},
        {"num_channels": [64, 128, 256], "kernel_size": 5, "dropout": 0.3},
    ]

    for idx, params in enumerate(tcn_experiments):
        experiment_name = f"TCN_Exp{idx+1}"
        logging.info(f"Starting Experiment: {experiment_name}")

        model = TemporalConvNet(
            num_inputs=257,  # 입력 채널
            num_channels=params["num_channels"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"],
            input_size=input_size,
        ).to(device)

        # 학습 및 검증
        train_losses, val_losses = train_and_validate(
            model,
            train_loader,
            val_loader,
            epochs=10,
            lr=1e-4,
            patience=5,
            experiment_name=experiment_name,
            tcn_params=params,
        )

        # Pruning 및 Fine-tuning
        prune_and_fine_tune(model, train_loader, val_loader)

        logging.info(f"Experiment {experiment_name} completed.")


if __name__ == "__main__":
    import os

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    main()
