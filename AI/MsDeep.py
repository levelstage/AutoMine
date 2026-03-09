import sys
import os
#  --- [Windows DLL  등록] ---
# MKL이 몰래 끌어다 쓰는 TBB, OpenMP 등의 경로를 모조리 등록합니다.
oneapi_root = r"C:\Program Files (x86)\Intel\oneAPI"
print("의존성 DLL 경로 등록 중...")

# 현재 스크립트 위치 기준으로 build 폴더를 sys.path에 추가 (모듈 검색용)
sys.path.append(os.path.join(os.path.dirname(__file__), 'build'))
sys.path.append(os.path.dirname(__file__))

# 인텔 주요 라이브러리 경로들
extra_dll_paths = [
    r"compiler\latest\include",
    r"compiler\latest\bin",
    r"mkl\latest\include",
    r"mkl\latest\bin",\
    r"tbb\latest\include",
    r"tbb\latest\bin",       # <-- 이게 없어서 터졌을 확률이 매우 높습니다 (스레딩 빌딩 블록)
    r"compiler\latest\windows\redist\intel64_win\compiler", # OpenMP (libiomp5md.dll)
    r"mkl\latest\redist\intel64",
    r"compiler\latest\windows\bin"
]

for sub in extra_dll_paths:
    p = os.path.join(oneapi_root, sub)
    if os.path.exists(p):
        try:
            os.add_dll_directory(p)
            print(f"  -> 등록 완료: {p}")
        except Exception:
            pass
print("-" * 50)

from Networks.network import DeepConvNet
from Networks.optimizer import Adam
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
iters = 2000
batch_size = 256
learning_rate = 0.001

# 각각의 기본값을 미리 맞춰 두었으므로 그대로 사용.
net = DeepConvNet()
trainer = MsTeacher()
opt = Adam(lr=learning_rate)

# 이전에 학습했던 my_model.pkl 얘를 불러옴
# net.load_params()

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
    grad = net.gradient(x.astype(np.float32), t.astype(np.float32))
    
    # optimizer로 가중치 갱신!
    opt.update(net, grad)

    # 손실함수값을 한 에폭마다 보여준다.
    loss = net.loss(x.astype(np.float32), t)
    print(str(i//epoch_size)+"번째 에폭, 현재 손실: " + str(loss))
net.save_params()
    