from Networks.network import DeepConvNet
from Teacher.teacher import MsTeacher
import numpy as np

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

def visualize_test(net, trainer):
    print("====== [AI 청문회 시작] ======")
    
    # 1. 테스트용 데이터 1개만 생성
    x_raw, t = trainer.generate_dataset(1) # batch_size=1
    x_raw = np.array(x_raw)
    t = np.array(t).reshape(1, -1) # (1, 100)
    
    # 2. 전처리 (One-hot)
    x = to_one_hot(x_raw) # (1, 10, 10, 10)
    
    # 3. AI 예측
    # predict 결과는 Logit(점수)이므로 Sigmoid를 씌워 확률로 변환
    logit = net.predict(x)
    pred_prob = logit.reshape(1, -1) # (1, 100)
    
    # 4. 마스크 생성 (닫힌 칸인 '0' 찾기)
    # x_raw는 (1, 10, 10) 형태. 0인 곳이 닫힌 칸.
    mask = (x_raw == 0).reshape(1, -1)
    
    # 5. 시각화
    # 10x10 판을 돌면서 '닫힌 칸'에 대해서만 물어봅니다.
    cnt = 0
    print("\n[게임판 상황 (0:닫힘, 1~9:힌트)]")
    print(x_raw[0])
    print("-" * 30)
    print(f"{'좌표':^10} | {'선생님(정답)':^15} | {'AI(예측)':^15} | {'판단'}")
    print("-" * 50)
    # 닫힌 칸 좌표만 뽑아서 비교
    indices = np.where(mask[0])[0]
    
    for idx in indices:
        row = idx // 10
        col = idx % 10
        
        target = t[0][idx]      # 정답 확률
        prediction = pred_prob[0][idx] # 예측 확률
        
        # 판단 로직 (오차 0.2 이내면 '정답' 인정)
        diff = abs(target - prediction)
        if diff < 0.2:
            judge = "✅ 정답"
        elif diff < 0.4:
            judge = "⚠️ 애매"
        else:
            judge = "❌ 땡!"
            
        print(f"({row}, {col}):  {target:.4f}   vs   {prediction:.4f}    {judge}")
        
        cnt += 1
            
    print("-" * 50)
    print("테스트 종료.")

net = DeepConvNet()
trainer = MsTeacher()
net.load_params()
# 실행
visualize_test(net, trainer)