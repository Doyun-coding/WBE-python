from fastapi import FastAPI
from pydantic import BaseModel
from server.session_manager import start_stt_session

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


# FastAPI 사용
app = FastAPI()


# Request 바디 구조 정의
# summoner_id 필드 요구
class SessionStartRequest(BaseModel):
    summoner_id: str


# POST 요청을 처리, 요청 바디는 SessionStartRequest 형태로 받는다
@app.post("/start-stt")
def start_session(req: SessionStartRequest):
    start_stt_session(req.summoner_id)

    # 요청을 보낸 클라이언트에 성공 응답 반환
    return {"status": "recording started"}
