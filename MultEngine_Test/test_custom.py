import os
import sys
import numpy as np
import time

# --- [Windows DLL  등록] ---
# MKL이 몰래 끌어다 쓰는 TBB, OpenMP 등의 경로를 모조리 등록합니다.
oneapi_root = r"C:\Program Files (x86)\Intel\oneAPI"
print("의존성 DLL 경로 등록 중...")

# 현재 스크립트 위치 기준으로 build 폴더를 sys.path에 추가 (모듈 검색용)
sys.path.append(os.path.join(os.path.dirname(__file__), 'build'))
sys.path.append(os.path.dirname(__file__))

# 인텔 주요 라이브러리 경로들
extra_dll_paths = [
    r"compiler\latest\bin",
    r"mkl\latest\bin",
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