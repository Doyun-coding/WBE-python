import tempfile, os, requests
import whisper
import numpy as np
from openai import OpenAI
from scipy.io import wavfile
from worker.stt.util.stt_worker_util import record_triggered_by_voice
from worker.tts.tts_worker_process import run_tts_worker
from threading import Thread

# OpenAI api key
client = OpenAI()
# whisper large (base, medium, large 클 수록 성능이 높음) 모델 로딩
model = whisper.load_model("base")


def load_prompt_template(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# whisper 사용해 stt 기능 함수
def whisper_pipeline(summoner_id, audio_data):
    print(f"[🔊 Whisper] {summoner_id} 음성 분석 시작")

    # 녹음된 오디오 데이터를 WAV 파일로 임시 저장
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        wavfile.write(tmpfile.name, 16000, (audio_data * 32768).astype(np.int16))
        path = tmpfile.name

    # Whisper 사용해서 음성을 텍스트로 변환
    result = model.transcribe(path)
    raw_text = result["text"]
    os.remove(path)

    print(f"[raw_text] : {raw_text}")

    # 텍스트를 게임 스타일로 정제하는 GPT 프롬프트
    prompt_template = load_prompt_template("prompt/champion_spell_prompt.txt")
    prompt = prompt_template.format(raw_text=raw_text)

    # GPT-4 API 호출
    gpt_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100
    )

    # 정제된 텍스트 결과 추출
    final_text = gpt_response.choices[0].message.content.strip()
    print(f"[🎯 결과] {summoner_id}: {final_text}")

    # 결과를 Spring 서버에 전송
    response = requests.post("http://localhost:8080/api/spell/flash", json={
        "summonerId": summoner_id,
        "finalText": final_text,
        "region": "KR"
    })

    if response.status_code == 201:
        print("✅ Created: 서버에서 자원을 성공적으로 생성함")

        run_tts_worker(summoner_id, response.text)
    else:
        print(f"❌ 실패 또는 다른 상태: {response.status_code} - {response.text}")


# Whisper 처리 프로세스
def run_process_worker(summoner_id):
    print(f"🔁 {summoner_id} 프로세스 실행 ")

    # 다음 음성 인식을 위한 새 프로세스를 미리 생성 (재귀 아님)
    def spawn_next():
        from multiprocessing import Process

        print(f"{summoner_id} 다음 프로세스 생성")
        Process(target=run_process_worker, args=(summoner_id,)).start()

    Thread(target=spawn_next, daemon=True).start()

    # 음성 인식 시작
    audio_data = record_triggered_by_voice()

    if len(audio_data) > 0:
        whisper_pipeline(summoner_id, audio_data)

    print(f"[🛑 종료] {summoner_id} 프로세스 종료")

