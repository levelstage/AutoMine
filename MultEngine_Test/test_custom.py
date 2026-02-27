import os
import sys
import numpy as np
import time

import sycl_mat

size = 4096
iterations = 2

print(f"[Custom SYCL] VTune 프로파일링 단독 실행 ({size}x{size}, {iterations}회)")

# Warm-up (디바이스 초기화 오버헤드를 VTune 기록에서 빼기 위함)
A_warm = np.random.rand(size, size).astype(np.float32)
B_warm = np.random.rand(size, size).astype(np.float32)
_ = sycl_mat.matmul(A_warm, B_warm)
print("Warm-up 완료.")

# 본 연산
for i in range(iterations):
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    
    start = time.perf_counter()
    _ = sycl_mat.matmul(A, B)
    end = time.perf_counter()
    print(f"  -> {i+1}회차: {end - start:.4f} 초")

print("프로파일링 종료!")