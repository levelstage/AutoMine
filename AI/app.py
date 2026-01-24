import sys
import numpy as np
from Networks.network import DeepConvNet

GAME_WIDTH = 10
GAME_HEIGHT = 10
GAME_MINE_COUNT = 10

def to_one_hot(grid_batch, num_classes=10):
    """
    grid_batch: (N, H, W) - 값은 0~10 정수
    Return: (N, C, H, W) - C=10
    """
    # 0 9개 1 1개 있는 벡터 10개를 만들고, grid_batch를 인덱스 배열로 사용해서 해당하는 위치에 벡터를 넣는다.
    # 형상은 (N, H, W, num_classes) 이렇게 된다.
    one_hot = np.eye(num_classes)[grid_batch]
    # 내가 원하는 형상은 (N, num_classes(C), H, W)이므로 transpose 하여 return
    return one_hot.transpose(0, 3, 1, 2)

def main():
    # 프로세스가 켜졌을 때 초기 정보를 뱉는다.
    print(f"{GAME_WIDTH} {GAME_HEIGHT} {GAME_MINE_COUNT}")
    sys.stdout.flush()
    net = DeepConvNet(input_dim=(10, GAME_WIDTH, GAME_HEIGHT))
    net.load_params("C:\\Junsu\\DLStudy\\AutoMine\\AI\\my_model.pkl")
    # 무한 루프로 C#의 명령을 기다림
    while True:
        board = []
        # 현재 판 상태 입력
        for _ in range(GAME_HEIGHT):
            line = sys.stdin.readline()
            if not line:
                return
            # 한 줄의 숫자들을 리스트로 변환
            row = list(map(int, line.split()))
            board.append(row)
        # board를 ndarray로 변환
        board = np.array(board).reshape((1, GAME_HEIGHT, GAME_WIDTH))
        # one-hot 인코딩
        x = to_one_hot(board)
        # 순전파(추론)
        y = net.predict(x)
        # 추론값을 표준출력으로 보낸다.
        probs = y.reshape(-1) 

        # 리스트 컴프리헨션으로 문자열 리스트를 만든 뒤, 한 번에 join
        output = " ".join([f"{p:.3f}" for p in probs])
        print(output)
        sys.stdout.flush()

if __name__ == "__main__":
    main()