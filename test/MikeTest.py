import sounddevice as sd
import numpy as np

# 마이크 테스트를 위한 클래스


def callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    print("📢 입력 레벨:", volume_norm)


with sd.InputStream(callback=callback):
    sd.sleep(5000)
