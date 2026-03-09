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

# 2. 모듈 임포트
import sycl_mat
import mkl_mat  # 새로 빌드한 MKL 모듈 추가

def benchmark(size, iterations=10):
    print(f"\n{'='*50}")
    print(f"대결 시작: 행렬 크기 {size} x {size} ({iterations}회 평균 측정)")
    print(f"{'='*50}")
    
    # --- Warm-up ---
    A_warm = np.random.rand(size, size).astype(np.float32)
    B_warm = np.random.rand(size, size).astype(np.float32)
    
    res_np_warmup = np.dot(A_warm, B_warm)
    res_sycl_warmup = sycl_mat.matmul(A_warm, B_warm)
    res_mkl_warmup = mkl_mat.matmul(A_warm, B_warm) # MKL 웜업 추가

    # 정합성 검증 (오차 허용 범위 1e-3)
    if not np.allclose(res_np_warmup, res_sycl_warmup, atol=1e-3):
        print("[경고] Custom SYCL 결과가 일치하지 않습니다. 로직 확인 필요.")
        return
    if not np.allclose(res_np_warmup, res_mkl_warmup, atol=1e-3):
        print("[경고] MKL SYCL 결과가 일치하지 않습니다.")
        return

    # --- 1. NumPy 연산 (CPU) ---
    np_times = []
    for _ in range(iterations):
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        start = time.perf_counter()
        _ = np.dot(A, B)
        end = time.perf_counter()
        np_times.append(end - start)
    
    avg_np_time = sum(np_times) / iterations
    print(f"NumPy (CPU) 평균      : {avg_np_time:.4f} 초")

    # --- 2. Custom SYCL 연산 (iGPU) ---
    sycl_times = []
    for _ in range(iterations):
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        start = time.perf_counter()
        _ = sycl_mat.matmul(A, B)
        end = time.perf_counter()
        sycl_times.append(end - start)
        
    avg_sycl_time = sum(sycl_times) / iterations
    print(f"Custom SYCL (iGPU) 평균: {avg_sycl_time:.4f} 초")

    # --- 3. oneMKL SYCL 연산 (iGPU) ---
    mkl_times = []
    for _ in range(iterations):
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        start = time.perf_counter()
        _ = mkl_mat.matmul(A, B)
        end = time.perf_counter()
        mkl_times.append(end - start)
        
    avg_mkl_time = sum(mkl_times) / iterations
    print(f"oneMKL SYCL (iGPU) 평균: {avg_mkl_time:.4f} 초")

    # --- 최종 스코어 보드 ---
    print("-" * 50)
    print(f"연산 결과 일치 검증 완료!")
    print(f"[NumPy vs Custom] Custom이 {avg_np_time / avg_sycl_time:.2f}배 빠름")
    print(f"[NumPy vs MKL]    MKL이 {avg_np_time / avg_mkl_time:.2f}배 빠름")
    print(f"[Custom vs MKL]   MKL이 {avg_sycl_time / avg_mkl_time:.2f}배 빠름")
    print("-" * 50)

# 메모리 할당 병목을 막기 위해 작은 크기는 10번, 4096은 5번만 측정합니다.
for s in [512, 1024, 2048]:
    benchmark(s, iterations=10)
benchmark(4096, iterations=5)