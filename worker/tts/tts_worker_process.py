import os
from openai import OpenAI

# GPT API KEY
client = OpenAI()


# tts 기능
def run_tts_worker(summoner_id, spell_text):
    # 음성 출력 파일 경로
    output_path = f"mp3/{summoner_id}_output.mp3"

    try:
        # tts 요청 커스텀
        tts = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=spell_text
        )

        # mp3 디렉토리 없으면 디렉토리 생성
        os.makedirs("mp3", exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(tts.content)

        os.system(f"afplay {output_path}")

    finally:
        # 재생 후 삭제
        if os.path.exists(output_path):
            os.remove(output_path)
