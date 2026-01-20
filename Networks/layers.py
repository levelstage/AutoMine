import numpy as np
import math
from .utils import col2im, im2col

class ReLU:
    def __init__(self):
        # 역전파 때 0 이하인 인덱스를 기억하기 위한 변수
        self.mask = None

    def forward(self, x):
        # 0 이하면 true, 아니면 false를 기억해 두었다가 역전파 때 활용
        self.mask = (x <= 0)
        # 원본 보존을 위해 복사
        out = x.copy()
        # Relu 연산
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # 상류의 미분값을 복사해서
        dx = dout.copy()
        # 0 이하였던 부분을 0으로 눌러버림 (나머지는 그대로)
        dx[self.mask] = 0
        return dx


class Sigmoid:
    def __init__(self):
        # 역전파 계산을 위해 순전파의 출력값(y)을 저장할 변수
        self.out = None

    def forward(self, x):
        # 시그모이드 공식에 x를 그대로 대입
        out = 1 / (1+np.exp(-x))
        # 역전파 계산을 위해 담아둠
        self.out = out
        return out

    def backward(self, dout):
        # 상류에서 들어온 미분값을 x에 대한 미분값으로 바꾼다.
        # 시그모이드의 dy/dx = y(1-y)
        dx = dout * self.out * (1 - self.out)
        return dx
    
class Conv2d:
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0, init_scale='he'):
        """
        init_scale: 'he' (ReLU용) 또는 'xavier' (Sigmoid용) 또는 float값
        """
        # 소중한 형상 정보들 저장
        self.stride = stride
        self.pad = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size

        # 입력 노드 수 계산
        fan_in = in_channels * filter_size * filter_size
        # 표준편차 계산
        if isinstance(init_scale, float):
            # 표준편차를 직접 정하는 경우
            z = init_scale
        elif init_scale == 'he':
            # he
            z = math.sqrt(2/fan_in)
        else:
            # xavier 
            # 사실 'xavier'라고 굳이 안치고 float가 아닌 값 아무거나 넣어도 닿음. ('he' 제외)
            z = math.sqrt(1/fan_in)
        
        # 가중치, 바이어스 초기값 설정
        self.W = np.random.randn(out_channels, in_channels, filter_size, filter_size) * z
        self.b = np.zeros(out_channels)
        
        # 역전파용 캐시
        self.x = None
        self.col = None
        self.col_W = None
        
        # 기울기 저장소
        self.dW = None
        self.db = None

    def forward(self, x):
        # 형상 정보 가져오기
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2*self.pad - FW) // self.stride + 1
        # 입력 데이터 전개
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 입력 데이터에 곱할 가중치도 전개
        col_W = self.W.reshape(FN, -1).T
        # 합성곱 연산! 이거 한번 하려고 지금까지 행렬을 접었다가 폈다가, 왔다리 갔다리...
        out = np.dot(col, col_W) + self.b
        # 출력에 맞게 형상을 변환시켜 준다.
        # (N*OH*OW, FN) => (N, OH, OW, FN)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        # 출력 전에 역전파를 대비해 소중한 변수들을 저장
        self.x = x
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, dout):
        # dout을 (N*OH*OW, FN) 형상으로 바꿔준다.
        dout = dout.transpose(0,2,3,1).reshape(-1, self.out_channels)
        # dL/db = dL/dy @ dy/db(=1), but dL/db의 형상이 (N*OH*OW, FN)이므로 전부 합산해서 (FN)으로 눌러줌.  
        self.db = np.sum(dout, axis=0)
        # dW(dL/dW) = dL/dy * dy/dW = x.T @ dout 
        self.dW = np.dot(self.col.T, dout).T.reshape(self.out_channels, self.in_channels, self.filter_size, self.filter_size)
        # dx(dL/dx) = dL/dy * dy/dx = dout @ W.T, 형상을 원상복구해주면서 return 하면 끝! 
        dx = col2im(np.dot(dout, self.col_W.transpose()), self.x.shape, self.filter_size, self.filter_size, self.stride, self.pad)
        return dx
    
class Affine:
    """
    init_scale: 'he' (ReLU용) 또는 'xavier' (Sigmoid용) 또는 float값
    """
    def __init__(self, input_size, output_size, init_scale='he'):
        # 입력/출력층 크기 입력
        self.input_size = input_size
        self.output_size = output_size

        # 실수값이 들어오면 그대로 쓰고, 아닐 경우 he 또는 xavier 사용
        fan_in = input_size
        if isinstance(init_scale, float):
            z = init_scale
        elif init_scale == 'he':
            z = math.sqrt(2/fan_in)
        else:
            z = math.sqrt(1/fan_in)

        # 파라미터를 담아둘 변수
        self.W = np.random.randn(input_size, output_size) * z
        self.b = np.zeros(output_size)

        # 미분값을 담아둘 변수
        self.dW = None
        self.db = None

        # 역전파를 대비해 x를 저장해둠.
        self.x = None
    def forward(self, x):
        # backward를 대비해 x를 미리 담아둔다
        self.x = x
        # 형상 변환 (N, C, H, W) => (N, input_size)
        x = x.reshape(-1, self.input_size)
        # (N, input_size) @ (input_size, output_size) => (N, output_size)
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):
        # dL/db = dL/dy @ dy/db = dL/dy = dout
        self.db = np.sum(dout, axis=0)
        # dL/dW = dL/dy * dy/dW = x.T @ dout
        # (input_size, N) @ (N, output_size) => (input_size, output_size)
        self.dW = np.dot(self.x.reshape(-1, self.input_size).T, dout)
        # dL/dx = dL/dy * dy.dx = dout @ W.T
        # (N, output_size) * (output_size, input_size) => 원본 모양으로 reshape
        dx = np.dot(dout, self.W.T).reshape(self.x.shape)
        return dx
    
class BinaryCrossEntropy:
    def __init__(self):
        self.y = None
        self.t = None
        self.mask = None
        self.weights = None
        
    def forward(self, y, t, mask, weights):
        """
        y: 신경망의 출력 (확률값, 0.0 ~ 1.0) - Shape: (N, 1) 또는 (N,)
        t: 정답 레이블 (확률값, 0.0 ~ 1.0) - Shape: y와 동일
        """
        self.y = y
        self.t = t
        self.mask = mask
        self.weights = weights

        #batch_size 가져오기
        batch_size = y.shape[0]
        
        # 로그 폭발 방지용 작은 값 (예: 1e-7)
        delta = 1e-7
        
        # 배치 전체의 평균 에러 계산
        # 공식: -sum( t*log(y) + (1-t)*log(1-y) ) / batch_size
        loss = -(t * np.log(y+delta) + (1-t) * np.log(1-y+delta)) * mask
        if weights is not None:
            loss = loss * weights
        return np.sum(loss) / np.sum(mask)

    def backward(self, dout=1):
        """
        dout: 상류에서 온 미분값 (보통 1)
        Returns: dx (입력 y에 대한 미분값 dL/dy)
        """
        # batch_size 가져오기
        batch_size = self.y.shape[0]
        delta = 1e-7
        
        # dL/dy 계산
        # 미분 공식: (y - t) / (y * (1 - y))
        # dL/dy = d forward(y, t) / dy 와 같으므로 dout은 상수로 사용할 수 있음.
        dx = self.mask * ((self.y - self.t) / (self.y * (1-self.y) + delta)) * dout
        if self.weights is not None:
            dx = dx * self.weights
        return dx / np.sum(self.mask)