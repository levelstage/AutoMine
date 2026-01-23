import sys
import json
from Networks.network import DeepConvNet

def main():
    # 무한 루프로 C#의 명령을 기다림
    while True:
        try:
            # 1. C#에서 보낸 데이터 한 줄 읽기 (Blocking)
            line = sys.stdin.readline()
            if not line: break # 연결 끊김

            # 2. JSON 파싱
            request = json.loads(line)
            
            # --- [AI 로직 처리 구간] ---
            # board = request['board']
            # result = AutoMineSolver.solve(board)
            # -------------------------

            # 더미 데이터 (테스트용)
            response = {
                "status": "success",
                "message": "AI가 보드를 분석했습니다.",
                "teacher_prob": [[0.1, 0.9], [0.2, 0.8]] # 예시
            }

            # 3. 결과 전송 (print 후 반드시 flush!)
            print(json.dumps(response))
            sys.stdout.flush()

        except Exception as e:
            # 에러 발생 시 로그 전송
            err_msg = {"status": "error", "message": str(e)}
            print(json.dumps(err_msg))
            sys.stdout.flush()

if __name__ == "__main__":
    main()