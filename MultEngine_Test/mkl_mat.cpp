#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp> // 인텔 oneMKL 헤더
#include <stdexcept>

namespace py = pybind11;

py::array_t<float> mkl_matmul(py::array_t<float> a, py::array_t<float> b)
{
    // 싱글턴 큐 유지 (기존 코드와 동일)
    static sycl::queue q(sycl::default_selector_v);
    static bool has_system_usm = q.get_device().get_info<sycl::info::device::usm_system_allocations>();
    
    // 버퍼 정보 요청 및 차원/크기 검사 (기존 코드와 동일)
    py::buffer_info pufA = a.request();
    py::buffer_info pufB = b.request();
    
    if(pufA.ndim != 2 || pufB.ndim != 2){
        throw std::runtime_error("Dimension Error: mkl_mat is made only for 2-dim matrices.");
    }
    if(pufA.shape[1] != pufB.shape[0]){
        throw std::runtime_error("Size Error: Sizes of the matrices are not suitable.");
    }
    
    const int M = pufA.shape[0], N = pufB.shape[1], K = pufA.shape[1];
    py::array_t<float> res({M, N});
    py::buffer_info pufR = res.request();
    
    float* ptrA = static_cast<float*>(pufA.ptr);
    float* ptrB = static_cast<float*>(pufB.ptr);
    float* ptrR = static_cast<float*>(pufR.ptr);

    float *d_A = ptrA, *d_B = ptrB, *d_R = ptrR;

    // USM 지원 여부에 따른 디바이스 메모리 할당 및 복사 (기존 코드와 동일)
    if(!has_system_usm)
    {
        d_A = sycl::malloc_device<float>(M * K, q);
        d_B = sycl::malloc_device<float>(K * N, q);
        d_R = sycl::malloc_device<float>(M * N, q);

        q.memcpy(d_A, ptrA, M * K * sizeof(float)).wait();
        q.memcpy(d_B, ptrB, K * N * sizeof(float)).wait();
    }

    // ---< MKL 연산 파트 >---
    try {
        // oneapi::mkl::blas::row_major::gemm 호출
        // 인자: queue, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc
        oneapi::mkl::blas::row_major::gemm(
            q, 
            oneapi::mkl::transpose::nontrans, 
            oneapi::mkl::transpose::nontrans, 
            M, N, K, 
            1.0f,       // alpha
            d_A, K,     // A의 선행 차원(lda)은 K
            d_B, N,     // B의 선행 차원(ldb)은 N
            0.0f,       // beta
            d_R, N      // C의 선행 차원(ldc)은 N
        ).wait();       // 연산이 끝날 때까지 대기
    } 
    catch(sycl::exception const& e) {
        // 예외 발생 시 할당된 디바이스 메모리를 먼저 정리 (USM 미지원 환경)
        if (!has_system_usm) {
            sycl::free(d_A, q);
            sycl::free(d_B, q);
            sycl::free(d_R, q);
        }
        throw std::runtime_error(std::string("oneMKL Exception: ") + e.what());
    }

    // 사용한 메모리 해제 및 복사 (기존 코드와 동일)
    if(!has_system_usm)
    {
        q.memcpy(ptrR, d_R, M * N * sizeof(float)).wait();
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_R, q);
    }
    return res;
}

PYBIND11_MODULE(mkl_mat, m) {
    m.def("matmul", &mkl_matmul, "oneMKL based Matrix Multiplication via SYCL");
}