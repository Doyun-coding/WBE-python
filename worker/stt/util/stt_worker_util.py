import webrtcvad
import collections
import sounddevice as sd
import numpy as np

# VAD 설정
vad = webrtcvad.Vad(3)

# 녹음 설정
sample_rate = 16000         # 오디오 샘플링 주파수 (16kHz)
frame_duration = 30         # 1 프레임 길이 (ms)
frame_size = int(sample_rate * frame_duration / 1000)   # 30ms 해당하는 샘플 수
channels = 1                # 채널 1개 (모노)
silence_threshold = 33      # 약 1초간 무음이면 종료 (33 x 30ms)


# 바이트 데이터를 음성인지 여부를 판단하는 함수
def is_speech(frame_bytes):
    return vad.is_speech(frame_bytes, sample_rate)


# 실제 음성 감지를 기반으로 녹음을 수행하는 함수
def record_triggered_by_voice():
    print("🟢 대기 중... (말하면 자동 녹음 시작)")

    ring_buffer = collections.deque(maxlen=silence_threshold)   # 사전 무음 저장 버퍼
    recording = []                  # 최종 녹음 결과
    triggered = False               # 음성 감지 후 녹음 중 여부
    silence_count = 0               # 연속 무음 카운트
    stop_flag = {"stop": False}     # 녹음 종료 여부

    # 오디오 콜백 함수: 실시간 음성 감지 및 녹음 제어
    def callback(indata, frames, time, status):
        nonlocal triggered, silence_count, recording
        # 현재 입력 오디오 프레임
        pcm = indata[:, 0]
        # 바이트로 변환 후 VADA 음성 여부 판단
        pcm_bytes = (pcm * 32768).astype(np.int16).tobytes()
        speech = is_speech(pcm_bytes)
        # 볼륨 기준 필터도 적용 (소리가 너무 작으면 무시)
        volume = np.max(np.abs(pcm))

        if speech and volume > 0.02:
            print("🎙️ 음성 감지 중...")
        else:
            print("🔈 무음 상태...")

        # 녹음 중인 경우
        if triggered:
            recording.append(pcm.copy())

            if not speech:
                silence_count += 1
                # 1초간 무음인 경우 종료
                if silence_count > silence_threshold:
                    print("🔇 무음 감지 → 녹음 종료")
                    stop_flag["stop"] = True
            else:
                # 음성이 다시 감지되면 초기화
                silence_count = 0
        # 아직 녹음 전
        else:
            ring_buffer.append(pcm.copy())
            if speech and volume > 0.02:
                print("🎤 음성 시작 → 녹음 시작")
                triggered = True
                recording.extend(ring_buffer)
                ring_buffer.clear()

    # 스트림 설정 및 실행
    stream = sd.InputStream(
        channels=channels,
        samplerate=sample_rate,
        blocksize=frame_size,
        dtype='float32',
        callback=callback
    )

    # 오디오 입력 스트림 열고, 콜백에서 stop_flag True 될 때까지 대기
    with stream:
        while not stop_flag["stop"]:
            sd.sleep(50)

    print("🛑 스트림 종료")

    # 최종 녹음된 오디오 데이터를 하나로 병합하여 반환
    return np.concatenate(recording)
