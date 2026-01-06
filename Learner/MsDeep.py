# 다른 디렉토리에 만들어진 클래스를 임포트해오기 위함.
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import network
from Teacher.teacher import MsTeacher

# 원활한 학습을 위해 숫자 힌트 데이터를 원-핫 인코딩한다.
# 여기서 0은 닫힌 칸, 1~9는 각각 0~8을 의미한다.
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

# 학습 구현 부분

# 하이퍼파라미터 설정
input_dim = (10, 10, 10) # (채널 10개, 10*10 그리드)
lr = 0.01  # 학습률 (Learning Rate)
epochs = 20  # 에폭 수
batch_size = 100  # 한 번에 학습할 데이터 양
data_size = 5000  # 한 에폭당 생성할 데이터 수

# 학습 데이터 생성기 객체 초기화
teacher = MsTeacher(width=10, height=10, num_mines=15)

# 네트워크 생성 (DeepConvNet)
# input_dim을 제외하고는 기본값 사용
net = network.DeepConvNet(input_dim)

# 학습 과정을 기록할 리스트
train_loss_list = []
accuracy_list = []

print("학습을 시작합니다...")

# 학습 루프
for epoch in range(epochs):
    
    # 데이터 생성 (이번 에폭에서 쓸 데이터)
    print(f"\n[Epoch {epoch+1}/{epochs}] 데이터 생성 중...")
    x_raw, t = teacher.generate_dataset(data_size)
    
    # 리스트를 numpy 배열로 변환
    x_raw = np.array(x_raw)
    t = np.array(t).reshape(data_size, -1) # 정답 레이블 형상을 맞춰준다
    
    # 입력 데이터 전처리 (One-Hot Encoding)
    x = to_one_hot(x_raw)
    
    # 데이터 개수 확인
    iter_per_epoch = max(data_size // batch_size, 1)
    
    total_loss = 0
    
    for i in range(iter_per_epoch):
        # 미니배치(Mini-batch) 뽑기
        # x와 t에서 batch_size만큼 잘라내서 batch_x, batch_t를 만듦
        batch_mask = np.random.choice(data_size, batch_size) # 랜덤 샘플링
        batch_x = x[batch_mask]
        batch_t = t[batch_mask]
        
        # 순전파 & 역전파
        # 학습에 사용할 기울기를 구한다.
        grads = net.gradient(batch_x, batch_t)
        
        # 가중치 업데이트 (경사하강법)
        # 파라미터(W, b)를 기울기(grads) 반대 방향으로 아주 조금(lr) 이동
        # 식: W = W - lr * dW
        i = 1
        for layer in net.layers.values():
            if hasattr(layer, 'W'): 
                layer.W -= lr * grads['W'+str(i)]
                layer.b -= lr * grads['b'+str(i)]
                i+=1

        # 손실값 기록 (모니터링용)
        loss = net.loss(batch_x, batch_t)
        total_loss += loss
    
    # 에폭 종료 후 결과 출력
    avg_loss = total_loss / iter_per_epoch
    train_loss_list.append(avg_loss)
    print(f"   -> Avg Loss: {avg_loss:.4f}")

    # 간단한 정확도 테스트 (0.5 기준)
    # 큰 의미는 없음.
    y = net.predict(x[:100]) # 100개만 테스트
    predict_mines = (y > 0.5).astype(int)
    answer_mines = (t[:100] > 0.5).astype(int)
    acc = np.mean(predict_mines == answer_mines)
    print(f"   -> Batch Accuracy (Sample): {acc * 100:.2f}%")

print("학습 완료!")

# 그래프 그리기
plt.plot(train_loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

def visualize_game(net, teacher):
    # 1. 테스트용 게임 1판 생성
    x_raw, t = teacher.generate_dataset(1)
    
    # 2. AI 예측
    x = to_one_hot(np.array(x_raw))
    y = net.predict(x) # y는 "안전할 확률" (1=Safe, 0=Mine)

    # 3. 데이터 형태 정리
    board = np.array(x_raw).reshape(10, 10)
    target_safety = np.array(t).reshape(10, 10) # 1=Safe, 0=Mine
    predict_safety = y.reshape(10, 10)          # 1=Safe, 0=Mine

    # -------------------------------------------------------------
    # [핵심 수정] 안전도(Safety)를 위험도(Mine Probability)로 뒤집기!
    # -------------------------------------------------------------
    predict_mine_prob = 1.0 - predict_safety  # 0.9 안전 -> 0.1 지뢰
    
    # 지뢰 판단 기준 (안전도가 0.5보다 낮으면 지뢰)
    predict_is_mine = (predict_safety < 0.5)
    
    # 실제 지뢰 위치 (정답이 0.5보다 낮으면(0이면) 지뢰)
    actual_is_mine = (target_safety < 0.5)
    
    # 4. 그림 그리기
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # [왼쪽] 실제 문제
    ax = axes[0]
    ax.set_title("Input Board")
    ax.imshow(board, cmap='Pastel1')
    for r in range(10):
        for c in range(10):
            val = int(board[r, c])
            if val > 0: 
                ax.text(c, r, str(val-1), ha='center', va='center', color='black', fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # [가운데] AI의 머릿속 (이제 빨간색 = 지뢰!)
    ax = axes[1]
    ax.set_title("AI's Perception (Red = Mine)")
    # 안전도가 아니라 '지뢰 확률'을 그립니다.
    im = ax.imshow(predict_mine_prob, cmap='Reds', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([]); ax.set_yticks([])

    # [오른쪽] 채점 결과
    ax = axes[2]
    ax.set_title("Result Analysis")
    ax.imshow(board, cmap='Greys', alpha=0.3)
    
    correct_count = 0
    total_mines = np.sum(actual_is_mine) # 지뢰 개수 세기
    
    for r in range(10):
        for c in range(10):
            is_mine = actual_is_mine[r, c]
            ai_said_mine = predict_is_mine[r, c]
            
            if is_mine and ai_said_mine:
                # 지뢰를 잘 찾음 (Green O)
                ax.text(c, r, "O", ha='center', va='center', color='green', fontsize=20, fontweight='bold')
                correct_count += 1
            elif not is_mine and ai_said_mine:
                # 멀쩡한 땅을 지뢰라 함 (Red X)
                ax.text(c, r, "X", ha='center', va='center', color='red', fontsize=20, fontweight='bold')
            elif is_mine and not ai_said_mine:
                # 지뢰를 못 찾음 (Blue ?) - 이게 진짜 위험!
                ax.text(c, r, "?", ha='center', va='center', color='blue', fontsize=20, fontweight='bold')
    
    ax.set_xlabel(f"Found {correct_count} / {total_mines} Mines")
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.show()

# 실행!
print("\n🔍 시각화 결과 출력 중...")
visualize_game(net, teacher)
