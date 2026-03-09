#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp> // 인텔 oneMKL 헤더
#include <stdexcept>

namespace py = pybind11;

// 싱글턴을 프리프로세서로 옮겨버리기

static sycl::queue q(sycl::default_selector_v);
static bool has_system_usm = q.get_device().get_info<sycl::info::device::usm_system_allocations>();

py::array_t<float> mkl_matmul(py::array_t<float> a, py::array_t<float> b, bool a_T, bool b_T)
{
    // 버퍼 정보 요청 및 차원/크기 검사 (전치를 고려해서 검사)
    py::buffer_info pufA = a.request();
    py::buffer_info pufB = b.request();
    
    if(pufA.ndim != 2 || pufB.ndim != 2){
        throw std::runtime_error("Dimension Error: mkl_mat is made only for 2-dim matrices.");
    }
    if(pufA.shape[!a_T] != pufB.shape[b_T]){
        throw std::runtime_error("Size Error: Sizes of the matrices are not suitable.");
    }
    
    // 전치 후의 size를 기준으로 하여 맞춘다.

    const int M = a_T ? pufA.shape[1] : pufA.shape[0]; 
    const int N = b_T ? pufB.shape[0] : pufB.shape[1];
    const int K = a_T ? pufA.shape[0] : pufA.shape[1];

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
        // 선행 차원은 전치 이전의 것을 넘겨야 함.
        oneapi::mkl::blas::row_major::gemm(
            q, 
            a_T ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, 
            b_T ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans, 
            M, N, K, 
            1.0f,       // alpha
            d_A, pufA.shape[1],     // A의 선행 차원(lda). 전치 여부와 상관 없이 원본 K를 넘김
            d_B, pufB.shape[1],     // B의 선행 차원(ldb). 역시 전치 여부와 상관 없이 원본 N 넘기기.
            0.0f,       // beta
            d_R, N      // C의 선행 차원(ldc)은 N. 이건 전치 후를 기준으로 하는게 맞음.
        ).wait();       // 연산이 끝날 때까지 대기
    } 
    catch(sycl::exception const& e) {
        // 예외 발생 시 할당된 디바이스 메모리를 먼저 정리 (USM 미지원 환경)
        if (!has_system_usm) {
            sycl::free(d_A, q);
            sycl::free(d_B, q);
            sycl::free(d_R, q);
        }
        // SYCL 비동기 에러 핸들링
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

py::array_t<float> mkl_im2col(py::array_t<float> im, int FH, int FW, int stride, int pad) {
    py::buffer_info pufIm = im.request();
    int N = pufIm.shape[0];
    int C = pufIm.shape[1];
    int H = pufIm.shape[2];
    int W = pufIm.shape[3];

    int OH = (H + 2 * pad - FH) / stride + 1;
    int OW = (W + 2 * pad - FW) / stride + 1;

    // 결과 행렬 크기: (N*OH*OW, C*FH*FW)
    py::array_t<float> col({N * OH * OW, C * FH * FW});
    float* col_ptr = static_cast<float*>(col.request().ptr);
    float* im_ptr = static_cast<float*>(pufIm.ptr);

    float *d_im = im_ptr, *d_col = col_ptr;

    if (!has_system_usm) {
        d_im = sycl::malloc_device<float>(im.size(), q);
        d_col = sycl::malloc_device<float>(col.size(), q);
        q.memcpy(d_im, im_ptr, im.size() * sizeof(float)).wait();
    }

    int col_width = C * FH * FW;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(col.size()), [=](sycl::id<1> idx) {
            int r = idx / col_width;
            int c_idx = idx % col_width;

            int n = r / (OH * OW);
            int oy = (r / OW) % OH;
            int ox = r % OW;

            int c = c_idx / (FH * FW);
            int fh = (c_idx / FW) % FH;
            int fw = c_idx % FW;

            // 실제 이미지 좌표 계산 (패딩 포함)
            int y = oy * stride + fh - pad;
            int x = ox * stride + fw - pad;

            // 범위 체크 (패딩 영역은 0)
            if (y >= 0 && y < H && x >= 0 && x < W) {
                d_col[idx] = d_im[((n * C + c) * H + y) * W + x];
            } else {
                d_col[idx] = 0.0f;
            }
        });
    }).wait();

    if (!has_system_usm) {
        q.memcpy(col_ptr, d_col, col.size() * sizeof(float)).wait();
        sycl::free(d_im, q);
        sycl::free(d_col, q);
    }

    return col;
}

py::array_t<float> mkl_col2im(py::array_t<float> col, int N, int C, int H, int W, int FH, int FW, int stride, int pad)
{
    int OH = (H + 2 * pad - FH) / stride + 1;
    int OW = (W + 2 * pad - FW) / stride + 1;
    int HP = H + 2 * pad; // Padded Height
    int WP = W + 2 * pad; // Padded Width

    py::array_t<float> res({N, C, HP, WP});
    float* res_ptr = static_cast<float*>(res.request().ptr);
    float* col_ptr = static_cast<float*>(col.request().ptr);

    float *d_col, *d_res;

    if(!has_system_usm) {
        // 디바이스 메모리 할당
        d_col = sycl::malloc_device<float>(col.size(), q);
        d_res = sycl::malloc_device<float>(res.size(), q);
        
        // 입력 복사 및 결과물 0 초기화 (필수!)
        q.memcpy(d_col, col_ptr, col.size() * sizeof(float));
        q.fill(d_res, 0.0f, res.size()).wait(); 
    } else {
        d_col = col_ptr;
        d_res = res_ptr;
        std::fill(d_res, d_res + res.size(), 0.0f);
    }

    // col 행렬의 가로 길이는 한 윈도우의 총 원소 수
    int col_width = C * FH * FW;

    q.submit([&](sycl::handler& h) {
        // 행렬의 모든 원소(N*OH*OW * C*FH*FW)를 병렬 처리
        h.parallel_for(sycl::range<1>(col.size()), [=](sycl::id<1> idx) {
            // 행렬 상의 row, col 위치 계산
            int r = idx / col_width;
            int c_idx = idx % col_width;

            // 윈도우 위치(n, oy, ox) 계산
            int n = r / (OH * OW);
            int oy = (r / OW) % OH;
            int ox = r % OW;

            // 필터 내 위치(c, fh, fw) 계산
            int c = c_idx / (FH * FW);
            int fh = (c_idx / FW) % FH;
            int fw = c_idx % FW;

            // 이미지 상의 실제 좌표 (Padding 포함 좌표)
            int y = oy * stride + fh;
            int x = ox * stride + fw;

            // 1차원 인덱스로 변환하여 Atomic Add
            int img_idx = ((n * C + c) * HP + y) * WP + x;
            
            auto atm = sycl::atomic_ref<float, sycl::memory_order::relaxed, 
                                       sycl::memory_scope::device, 
                                       sycl::access::address_space::global_space>(d_res[img_idx]);
            atm.fetch_add(d_col[idx]);
        });
    }).wait();

    if(!has_system_usm) {
        // 결과만 시스템 메모리로 가져오기
        q.memcpy(res_ptr, d_res, res.size() * sizeof(float)).wait();
        sycl::free(d_col, q);
        sycl::free(d_res, q);
    }

    return res;
}

PYBIND11_MODULE(mkl_mat, m) {
    m.def("matmul", &mkl_matmul, "oneMKL Matmul with Transpose", 
          py::arg("a"), py::arg("b"), py::arg("a_T") = false, py::arg("b_T") = false);
    m.def("im2col", &mkl_im2col, "GPU optimized im2col via SYCL");
    m.def("col2im", &mkl_col2im, "GPU optimized col2im via SYCL");
}