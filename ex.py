import webrtcvad
import collections
import sounddevice as sd
import numpy as np
import whisper
import os
import tempfile
import scipy.io.wavfile as wav
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI api key
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
# process 모델 설정
model = whisper.load_model("base")

# 음성 인식 민감도 2로 설정 / 0~3 (3이 가장 민감)
vad = webrtcvad.Vad(2)

# 녹음 파라미터
sample_rate = 16000     # 1초에 16000개의 오디오 샘플을 녹음
frame_duration = 30     # 오디오를 30ms 단위로 잘라서 분석
frame_size = int(sample_rate * frame_duration / 1000)   # 30ms 동안의 오디오를 샘플링하면 총 480개의 샘플이 생긴다는 의미
channels = 1            # 모노 채널
silence_threshold = 33  # 무음이 15프레임 연속(약 30 × 30ms = 1초) 감지되면 "녹음 끝났다"고 판단


def is_speech(frame_bytes):
    return vad.is_speech(frame_bytes, sample_rate)

# 사용자가 말하기 시작하면 자동 녹음 시작, 침묵이 이어지면 종료
def record_triggered_by_voice():
    print("🟢 대기 중... (말하면 자동 녹음 시작)")

    ring_buffer = collections.deque(maxlen=silence_threshold)
    recording = []
    triggered = False
    silence_count = 0

    stop_flag = {"stop": False}  # stop을 외부에서 제어하기 위한 플래그

    def callback(indata, frames, time, status):
        nonlocal triggered, silence_count, recording
        pcm = indata[:, 0]
        pcm_bytes = (pcm * 32768).astype(np.int16).tobytes()
        speech = is_speech(pcm_bytes)
        volume = np.max(np.abs(pcm))

        if speech and volume > 0.02:
            print("🎙️ 음성 감지 중...")
        else:
            print("🔈 무음 상태...")

        if triggered:
            recording.append(pcm.copy())
            if not speech:
                silence_count += 1
                if silence_count > silence_threshold:
                    print("🔇 무음 감지 → 녹음 종료")
                    stop_flag["stop"] = True  # 플래그로 종료 신호
            else:
                silence_count = 0
        else:
            ring_buffer.append(pcm.copy())
            if speech and volume > 0.02:
                print("🎤 음성 시작 → 녹음 시작")
                triggered = True
                recording.extend(ring_buffer)
                ring_buffer.clear()

    # 수동 스트림 제어 방식
    stream = sd.InputStream(channels=channels, samplerate=sample_rate,
                            blocksize=frame_size, dtype='float32',
                            callback=callback)
    with stream:
        while not stop_flag["stop"]:
            sd.sleep(50)  # 50ms 간격으로 stop 상태 확인

    print("🛑 스트림 종료")

    return np.concatenate(recording)

def main():
    while True:

        # 자동 녹음
        audio_data = record_triggered_by_voice()

        if len(audio_data) == 0:
            print("⚠️ 녹음된 데이터가 없습니다. 다시 대기합니다.")
            continue

        print("🟢 완료... (녹음 완료)")

        # 임시 wav 파일 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            wav.write(tmpfile.name, sample_rate, (audio_data * 32768).astype(np.int16))
            temp_filename = tmpfile.name

        print(f"📁 저장된 WAV 파일: {temp_filename} ({os.path.getsize(temp_filename)} bytes)")

        # Whisper 처리
        result = model.transcribe(temp_filename)
        raw_text = result["text"]
        print("🎧 Whisper:", raw_text)

        # GPT-4 정제
        prompt = f"""
        너는 리그 오브 레전드(LOL) 게임의 실시간 보이스 채팅을 텍스트로 바꿔주는 AI야.
        다음 문장은 음성 인식 결과인데, 이를 **LOL 게임 내 채팅 스타일**에 맞게 **간결하고 자연스럽게 정제**해줘.
        게임 내 용어 의미 전달이 확실하게 되도록 해줘.
        한글로 대답해줘
        입력: "{raw_text}"
        출력:
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 리그 오브 레전드(LOL) 게임의 실시간 보이스 채팅을 텍스트로 바꿔주는 AI야."
                                              "다음 문장은 음성 인식 결과인데, 이를 **LOL 게임 내 채팅 스타일**에 맞게 **간결하고 자연스럽게 정제**해줘."
                                              "게임 내 용어 의미 전달이 확실하게 되도록 해줘. 한글로 대답해줘"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        final_text = response.choices[0].message.content.strip()
        print("✨ 정제:", final_text)

        # TTS
        tts = client.audio.speech.create(model="tts-1", voice="nova", input=final_text)
        with open("mp3/output.mp3", "wb") as f:
            f.write(tts.content)
        os.system("afplay output.mp3")

        os.remove(temp_filename)


if __name__ == "__main__":
    main()
