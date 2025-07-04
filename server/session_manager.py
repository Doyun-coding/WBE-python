import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

# 쓰레드 풀 (동시 사용자 최대 20명까지 처리)
thread_pool = ThreadPoolExecutor(max_workers=200)

# 현재 실행 중인 사용자 세션을 추적
active_users = {}


# 사용자 세션을 시작하는 함수
def start_stt_session(summoner_id):

    # # 이미 같은 사용자의 세션이 실행 중인 경우 중복 실행 방지
    # if summoner_id in active_users:
    #     print(f"⚠️ {summoner_id} 세션 이미 존재")
    #     return

    # 실제 쓰레드에서 실행될 작업
    def thread_task():
        from worker.stt.stt_worker_process import run_process_worker

        print(f"[Thread Start] {summoner_id}")

        # 새로운 프로세스를 생성하여 음성 처리 루프 실행
        proc = Process(target=run_process_worker, args=(summoner_id,))
        proc.start()
        # 사용자 세션 등록
        active_users[summoner_id] = proc

        # # 프로세스가 살아있는 동안 대기
        while proc.is_alive():
            time.sleep(1)

        # 프로세스가 종료되면 사용자 세션에서 제거
        print(f"[🛑 종료] {summoner_id} 프로세스 종료")
        active_users.pop(summoner_id, None)
    # 위의 thread_task 를 thread_pool 에 제출하여 백그라운드에 실행
    # 쓰레드 사용이 종료되면 자동으로 쓰레드 풀로 반환
    thread_pool.submit(thread_task)

    print(f"✅ {summoner_id} 세션 등록됨")
