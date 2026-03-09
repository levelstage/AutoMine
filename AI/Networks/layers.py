import numpy as np
import time
import mkl_mat as mkl
import math

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
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
        f_total = time.perf_counter()
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2*self.pad - FW) // self.stride + 1
        
        # 1. im2col 구간
        t_im2col = time.perf_counter()
        # 데이터 타입을 float32로 맞춰서 MKL 에러 방지
        col = mkl.im2col(x, FH, FW, self.stride, self.pad).astype(np.float32)
        col_W = self.W.reshape(FN, -1) 
        time_im2col = time.perf_counter() - t_im2col
        
        # 2. Pure MKL 구간 (b_T=True 사용하여 col_W.T 복사 비용 제거)
        t_mkl = time.perf_counter()
        out = mkl.matmul(col, col_W, a_T=False, b_T=True) + self.b
        time_mkl = time.perf_counter() - t_mkl
        
        # 3. Reshape 구간
        t_res = time.perf_counter()
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        time_res = time.perf_counter() - t_res
        
        self.x = x
        self.col = col
        
        #print(f"[Conv2d Forward] Total: {time.perf_counter()-f_total:.4f}s | im2col: {time_im2col:.4f}s | MKL: {time_mkl:.4f}s | Reshape: {time_res:.4f}s")
        return out

    def backward(self, dout):
        b_total = time.perf_counter()
        FN, C, FH, FW = self.W.shape
        N, C, H, W = self.x.shape
        
        # 1. Prepare/Transpose 구간
        t_prep = time.perf_counter()
        dout_reshaped = np.ascontiguousarray(dout.transpose(0, 2, 3, 1).reshape(-1, FN), dtype=np.float32)
        self.db = np.sum(dout_reshaped, axis=0)
        time_prep = time.perf_counter() - t_prep

        # 2. Gradient W (MKL) - a_T=True 사용하여 col.T 복사 제거
        t_gw = time.perf_counter()
        # col(N*OH*OW, C*FH*FW).T @ dout(N*OH*OW, FN) -> (C*FH*FW, FN)
        # 결과물을 .T 해서 (FN, C*FH*FW)로 만든 뒤 원래 형상으로 복원
        self.dW = mkl.matmul(self.col, dout_reshaped, a_T=True, b_T=False).T.reshape(FN, C, FH, FW)
        time_gw = time.perf_counter() - t_gw
        
        # 3. Gradient X (MKL + col2im)
        t_gx = time.perf_counter()
        col_W = self.W.reshape(FN, -1)
        # dout_reshaped @ col_W -> dx_col(N*OH*OW, C*FH*FW)
        dx_col = mkl.matmul(dout_reshaped, col_W, a_T=False, b_T=False)
        
        # 우리가 만든 수제 GPU col2im 커널!
        dx_padded = mkl.col2im(dx_col, N, C, H, W, FH, FW, self.stride, self.pad)
        
        # 패딩 제거
        if self.pad > 0:
            dx = dx_padded[:, :, self.pad:-self.pad, self.pad:-self.pad]
        else:
            dx = dx_padded
        time_gx = time.perf_counter() - t_gx
        
        #print(f"[Conv2d Backward] Total: {time.perf_counter()-b_total:.4f}s | Prep: {time_prep:.4f}s | dW_MKL: {time_gw:.4f}s | dX(MKL+col2im): {time_gx:.4f}s")
        return dx

class Affine:
    def __init__(self, input_size, output_size, init_scale='he'):
        self.input_size = input_size
        self.output_size = output_size
        fan_in = input_size
        z = math.sqrt(2/fan_in) if init_scale == 'he' else math.sqrt(1/fan_in)
        if isinstance(init_scale, float): z = init_scale

        self.W = (np.random.randn(input_size, output_size) * z).astype(np.float32)
        self.b = np.zeros(output_size, dtype=np.float32)

        self.dW = None
        self.db = None
        self.x = None
        
    def forward(self, x):
        f_total = time.perf_counter()
        self.x = x
        x_reshaped = x.reshape(-1, self.input_size).astype(np.float32)
        # x_reshaped @ self.W -> (N, output_size)
        out = mkl.matmul(x_reshaped, self.W) + self.b
        #print(f"[Affine Forward] Total: {time.perf_counter()-f_total:.4f}s")
        return out
    
    def backward(self, dout):
        b_total = time.perf_counter()
        dout_contig = np.ascontiguousarray(dout, dtype=np.float32)
        self.db = np.sum(dout_contig, axis=0)
        
        x_reshaped = self.x.reshape(-1, self.input_size).astype(np.float32)
        # dW = x.T @ dout (a_T=True 활용)
        self.dW = mkl.matmul(x_reshaped, dout_contig, a_T=True)
        
        # dx = dout @ W.T (b_T=True 활용)
        dx = mkl.matmul(dout_contig, self.W, b_T=True).reshape(self.x.shape)
        #print(f"[Affine Backward] Total: {time.perf_counter()-b_total:.4f}s")
        return dx

class BinaryCrossEntropy:
    def __init__(self):
        self.y = None
        self.t = None
        self.mask = None
        self.weights = None
        
    def forward(self, y, t, mask, weights):
        self.y = y
        self.t = t
        self.mask = mask
        self.weights = weights

        delta = 1e-7
        loss = -(t * np.log(y+delta) + (1-t) * np.log(1-y+delta)) * mask
        if weights is not None:
            loss = loss * weights
        res = np.sum(loss) / np.sum(mask)
        return res

    def backward(self, dout=1.0):
        delta = 1e-7
        dx = self.mask * ((self.y - self.t) / (self.y * (1-self.y) + delta)) * dout
        if self.weights is not None:
            dx = dx * self.weights
        res = dx / np.sum(self.mask)
        return res