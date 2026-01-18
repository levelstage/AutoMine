from Networks.network import DeepConvNet
from Teacher.teacher import MsTeacher
import numpy as np

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

# 하이퍼파라미터 설정
epoch_size = 100  # 에폭은 그냥 편의상 나눠서 확인(데이터가 무한히 많음.)
iters = 10000
batch_size = 100
learning_rate = 0.01

# 각각의 기본값을 미리 맞춰 두었으므로 그대로 사용.
net = DeepConvNet()
trainer = MsTeacher()

print("학습 시작!")
for i in range(iters):
    print("학습 진행중... ("+str(i)+"/"+str(iters)+")")
    # 배치 크기만큼 학습 데이터를 생성
    x_raw, t = trainer.generate_dataset(batch_size)
    
    # 리스트를 numpy 배열로 변환
    x_raw = np.array(x_raw)
    t = np.array(t).reshape(batch_size, -1) # 정답 레이블 형상을 맞춰준다

    # x를 원-핫 인코딩
    x = to_one_hot(x_raw)

    # 기울기를 구해준다.
    grad = net.gradient(x, t)
    
    # 구한 기울기를 경사하강법으로 갱신
    j=1
    for layer in net.layers.values():
        if hasattr(layer, 'W'):
            layer.W -= learning_rate * grad['W'+str(j)]
            layer.b -= learning_rate * grad['b'+str(j)]
            j+=1
    # 손실함수값을 한 에폭마다 보여준다.
    loss = net.loss(x, t)
    if i % epoch_size == 0:
        print(str(i//epoch_size)+"번째 에폭, 현재 손실: " + str(loss))
net.save_params()
    