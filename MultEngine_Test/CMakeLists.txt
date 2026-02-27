cmake_minimum_required(VERSION 3.14)
project(sycl_mat_project)

# 1. 파이썬과 pybind11 환경 찾기
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ---------------------------------------------------
# [타겟 1] 직접 최적화한 커스텀 SYCL 모듈
# ---------------------------------------------------
pybind11_add_module(sycl_mat sycl_mat.cpp)

# Intel DPC++ (SYCL) 전용 컴파일 옵션 붙이기
target_compile_options(sycl_mat PRIVATE -fsycl -O3)
target_link_options(sycl_mat PRIVATE -fsycl)

# ---------------------------------------------------
# [타겟 2] oneMKL 기반 SYCL 모듈 (새로 추가)
# ---------------------------------------------------
pybind11_add_module(mkl_mat mkl_mat.cpp)

# MKL을 쓰기 위해 핵심 플래그인 '-Qmkl'을 추가합니다.
target_compile_options(mkl_mat PRIVATE -fsycl -Qmkl -O3)
target_link_options(mkl_mat PRIVATE -fsycl -Qmkl)