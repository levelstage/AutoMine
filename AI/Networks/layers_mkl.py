import numpy as np
import mkl_mat as mkl
import math

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # 역전파에서 기울기를 막을 위치를 기억해둔다
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # 순전파에서 0 이하였던 곳은 기울기를 0으로 막아준다
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        # 역전파 계산에 출력값이 필요하므로 저장해둔다
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)

class Conv2d:
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0, init_scale='he'):
        self.stride = stride
        self.pad = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size

        # 분산 폭발/소실을 막기 위해 He 초기화를 사용한다
        fan_in = in_channels * filter_size * filter_size
        if isinstance(init_scale, float):
            z = init_scale
        elif init_scale == 'he':
            z = math.sqrt(2/fan_in)
        else:
            z = math.sqrt(1/fan_in)

        self.W = (np.random.randn(out_channels, in_channels, filter_size, filter_size) * z).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)

        self.x = None
        self.col = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2*self.pad - FW) // self.stride + 1

        # im2col로 입력을 2D 행렬로 펼쳐서 합성곱을 행렬곱으로 처리한다
        col = mkl.im2col(x, FH, FW, self.stride, self.pad).astype(np.float32)
        col_W = self.W.reshape(FN, -1)
        out = mkl.matmul(col, col_W, a_T=False, b_T=True) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # 역전파에서 dW, dx 계산에 필요하므로 저장해둔다
        self.x = x
        self.col = col
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = self.x.shape

        # 행렬곱 형태로 맞추기 위해 dout을 (N*OH*OW, FN) 형상으로 변환한다
        dout_reshaped = np.ascontiguousarray(dout.transpose(0, 2, 3, 1).reshape(-1, FN), dtype=np.float32)
        self.db = np.sum(dout_reshaped, axis=0)

        # col.T @ dout으로 dW를 구한 뒤 필터 형상으로 복원한다
        self.dW = mkl.matmul(self.col, dout_reshaped, a_T=True, b_T=False).T.reshape(FN, C, FH, FW)

        # dout @ col_W로 dx_col을 구한 뒤 col2im으로 원래 형상으로 되돌린다
        col_W = self.W.reshape(FN, -1)
        dx_col = mkl.matmul(dout_reshaped, col_W, a_T=False, b_T=False)
        dx_padded = mkl.col2im(dx_col, N, C, H, W, FH, FW, self.stride, self.pad)

        # 순전파에서 추가했던 패딩을 제거한다
        if self.pad > 0:
            dx = dx_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
        else:
            dx = dx_padded
        return dx

class Affine:
    def __init__(self, input_size, output_size, init_scale='he'):
        self.input_size = input_size
        self.output_size = output_size

        # 분산 폭발/소실을 막기 위해 He 초기화를 사용한다
        fan_in = input_size
        z = math.sqrt(2/fan_in) if init_scale == 'he' else math.sqrt(1/fan_in)
        if isinstance(init_scale, float): z = init_scale

        self.W = (np.random.randn(input_size, output_size) * z).astype(np.float32)
        self.b = np.zeros(output_size, dtype=np.float32)

        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        # 역전파에서 dW 계산에 필요하므로 입력을 저장해둔다
        self.x = x
        x_reshaped = x.reshape(-1, self.input_size).astype(np.float32)
        out = mkl.matmul(x_reshaped, self.W) + self.b
        return out

    def backward(self, dout):
        dout_contig = np.ascontiguousarray(dout, dtype=np.float32)
        self.db = np.sum(dout_contig, axis=0)

        x_reshaped = self.x.reshape(-1, self.input_size).astype(np.float32)
        self.dW = mkl.matmul(x_reshaped, dout_contig, a_T=True)   # x.T @ dout
        dx = mkl.matmul(dout_contig, self.W, b_T=True).reshape(self.x.shape)  # dout @ W.T
        return dx

class BinaryCrossEntropy:
    def __init__(self):
        self.y = None
        self.t = None
        self.mask = None
        self.weights = None

    def forward(self, y, t, mask, weights):
        # 역전파에서 사용하기 위해 순전파 값들을 저장해둔다
        self.y = y
        self.t = t
        self.mask = mask
        self.weights = weights

        delta = 1e-7  # log(0) 방지
        loss = -(t * np.log(y+delta) + (1-t) * np.log(1-y+delta)) * mask
        if weights is not None:
            loss = loss * weights
        return np.sum(loss) / np.sum(mask)

    def backward(self, dout=1.0):
        delta = 1e-7
        dx = self.mask * ((self.y - self.t) / (self.y * (1-self.y) + delta)) * dout
        if self.weights is not None:
            dx = dx * self.weights
        return dx / np.sum(self.mask)
