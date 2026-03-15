// SYCL만을 이용해서 스스로 최적화해본 커널.
// 커밋 로그는 못남겼지만, 블로그에 가면 과정을 확인해 볼 수 있다.
// https://velog.io/@levelstage/AutoMinepp0 참고

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>
#include <stdexcept>

namespace py = pybind11;

// float8을 4, 16 등으로 바꾸면 청크 크기 조절 가능. 단 CHUNK_SIZE도 같이 조정해야 함.
typedef sycl::float8 float_chunk;
const int CHUNK_SIZE = 8;

// 연산에 필요한 상수들

// 기본 상수
const int TILE_M = 16, TILE_N = 16, TILE_K = 8;  // Tiling의 크기
const int BATCH_M = 8, BATCH_N = 4;  // Coarsening의 크기

// 파생 상수

// 한 워크 그룹에 들어가는 스레드의 수 (안씀)
// const int GROUP_THREAD_COUNT = TILE_M * TILE_N;

// 한 행에 들어가는 스레드의 수는 Tile의 열 방향 크기 / CHUNK_SIZE
const int A_ROW_THREAD_COUNT = TILE_K / CHUNK_SIZE;
const int B_ROW_THREAD_COUNT = TILE_N * BATCH_N / CHUNK_SIZE;

// 타일을 다 채우기 위해 필요한 쓰레드의 수. GROUP_THREAD_COUNT보다 작거나 같아야 한다.
const int A_TILE_THREAD_COUNT = TILE_M * BATCH_M * A_ROW_THREAD_COUNT;
const int B_TILE_THREAD_COUNT = TILE_K * B_ROW_THREAD_COUNT;

py::array_t<float> matmul(py::array_t<float> a, py::array_t<float> b)
{
    // 싱글턴 큐를 유지한다.
    static sycl::queue q(sycl::default_selector_v);
    static bool has_system_usm = q.get_device().get_info<sycl::info::device::usm_system_allocations>();
    // 두 ndArray a, b를 받아서, 각각의 buffer_info를 python에 요청한다.
    py::buffer_info pufA = a.request();
    py::buffer_info pufB = b.request();
    // 받은 buffer_info를 가지고 차원과 행렬의 크기를 검사한다.
    if(pufA.ndim != 2 || pufB.ndim != 2){
        throw std::runtime_error("Dimension Error: sycl_mat is made only for 2-dim matrices.");
    }
    if(pufA.shape[1] != pufB.shape[0]){
        throw std::runtime_error("Size Error: Sizes of the matrices are not suitable.");
    }
    // 통과했다면, 크기를 기록해두고, return을 위한 ndArray를 생성한 뒤, buffer_info를 가져온다.
    const int M = pufA.shape[0], N = pufB.shape[1], K = pufA.shape[1];
    py::array_t<float> res({M, N});
    py::buffer_info pufR = res.request();
    
    // 가져온 각각의 buffer_info에서 실제 데이터가 저장된 주소를 가져온다.
    // void* 형태로 시작 주소를 주기 때문에, float* 형태로 캐스팅
    float* ptrA = static_cast<float*>(pufA.ptr);
    float* ptrB = static_cast<float*>(pufB.ptr);
    float* ptrR = static_cast<float*>(pufR.ptr);

    // 이제 포인터를 넘겨받았으니, 그대로 사용한다.
    float *d_A = ptrA, *d_B = ptrB, *d_R = ptrR;

    // 만약, 시스템 메모리 영역을 그대로 쓰지 못하는 디바이스라면(ex: VRAM 달린 그래픽카드)
    // 직접 메모리 카피를 해준다.
    if(!has_system_usm)
    {

        // 디바이스 메모리에 영역을 확보해준다.(꼭 free해주자.)
        d_A = sycl::malloc_device<float>(M * K, q);
        d_B = sycl::malloc_device<float>(K * N, q);
        d_R = sycl::malloc_device<float>(M * N, q);

        // 디바이스 메모리로 필요한 데이터를 복사해준다.
        q.memcpy(d_A, ptrA, M * K * sizeof(float)).wait();
        q.memcpy(d_B, ptrB, K * N * sizeof(float)).wait();
    }

    q.submit([&](sycl::handler& h) {

        // Tiling을 위한 local memory 할당
        // double buffering을 위해 3차원으로 할당. (2 페이지)
        sycl::local_accessor<float, 3> tileA(sycl::range<3>(2, TILE_M*BATCH_M, TILE_K), h);
        sycl::local_accessor<float, 3> tileB(sycl::range<3>(2, TILE_K, TILE_N*BATCH_N), h);

        // float 배열을 float8 배열로 reinterpret. 사실 원래 float배열인데 float8로 보는 것이다.
        float_chunk* vec_A = reinterpret_cast<float_chunk*>(d_A);
        float_chunk* vec_B = reinterpret_cast<float_chunk*>(d_B);

        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(M/BATCH_M, N/BATCH_N), sycl::range<2>(TILE_M, TILE_N)), [=](sycl::nd_item<2> item) {
            // 우선 전체를 batch_size로 쪼개서 global id를 먼저 부여한 후,
            const int r = item.get_global_id(0), c = item.get_global_id(1);
            // 생성된 녀석들을 tile_size 단위로 묶어서 local id를 추가로 부여하는 방식.
            const int local_r = item.get_local_id(0), local_c = item.get_local_id(1);

            // 타일의 총 개수는 K / TILE_K
            const int num_tiles = K / TILE_K;
            
            // tid 계산 (이제 스레드를 일렬로 줄세운 후, 그대로 1차원 배열에 접근시킨다)
            const int tid = local_r * TILE_N + local_c;
            
            // 가져올 A와 B 청크의 global 좌표를 만들기 위한 base
            const int base_a = (r-local_r)*BATCH_M;
            const int base_b = (c-local_c)*BATCH_N/CHUNK_SIZE;

            // 가져올 A와 B 청크의 local 좌표
            const int load_a_r = tid/A_ROW_THREAD_COUNT;
            const int load_a_c = tid%A_ROW_THREAD_COUNT;
            const int load_b_r = tid/B_ROW_THREAD_COUNT;
            const int load_b_c = tid%B_ROW_THREAD_COUNT;
            
            // 외적값의 합을 저장할 배열을 생성
            float sum[BATCH_M][BATCH_N] = {};
            // 현재 읽어야 하는 페이지
            bool current_page = 0;

            // 가져온 청크를 해체하는 함수.
            auto split_A = [&](const bool &page, const float_chunk &chunk){
                #pragma unroll
                for(int i=0; i<CHUNK_SIZE; ++i) tileA[page][load_a_r][load_a_c*CHUNK_SIZE+i] = chunk[i];
            };

            auto split_B = [&](const bool &page, const float_chunk &chunk){
                #pragma unroll
                for(int i=0; i<CHUNK_SIZE; ++i) tileB[page][load_b_r][load_b_c*CHUNK_SIZE+i] = chunk[i];
            };

            // 우선 0페이지를 채워준다.
            float_chunk chunk_A, chunk_B;
            if(tid < A_TILE_THREAD_COUNT)
            {
                chunk_A = vec_A[(base_a + load_a_r) * (K / CHUNK_SIZE) + load_a_c];
                split_A(current_page, chunk_A);
            }
            if(tid < B_TILE_THREAD_COUNT)
            {
                chunk_B = vec_B[load_b_r * (N/CHUNK_SIZE) + (base_b + load_b_c)];
                split_B(current_page, chunk_B);
            }
            
            item.barrier(sycl::access::fence_space::local_space);

            // 타일링 루프 시작
            for(int t=0; t<num_tiles; ++t)
            {
                // ---< 연산 전에 미리 LSU에 chunk를 요청 > --- 
                if(t < num_tiles-1){
                    if(tid < A_TILE_THREAD_COUNT) chunk_A = vec_A[(base_a + load_a_r) * (K/CHUNK_SIZE) + (load_a_c + (t+1)*TILE_K/CHUNK_SIZE)];
                    if(tid < B_TILE_THREAD_COUNT) chunk_B = vec_B[(load_b_r + (t+1)*TILE_K) * (N/CHUNK_SIZE) + (base_b + load_b_c)];
                }
                
                // ---< 연산 파트 > ---
                float regA[BATCH_M], regB[BATCH_N];
                // K축으로 루프 시작
                #pragma unroll
                for(int k=0; k<TILE_K; ++k)
                {
                    // 로컬메모리 -> 레지스터로 ( ,k) / (k, ) 벡터를 옮긴다.
                    #pragma unroll
                    for (int i=0; i<BATCH_M; ++i) regA[i] = tileA[current_page][local_r*BATCH_M+i][k];
                    #pragma unroll
                    for (int i=0; i<BATCH_N; ++i) regB[i] = tileB[current_page][k][local_c*BATCH_N+i];
                    // 이제 오직 레지스터에만 접근하면서 외적을 계산할 수 있음!
                    #pragma unroll
                    for (int i=0; i<BATCH_M; ++i)
                    {
                        for(int j=0; j<BATCH_N; ++j)
                        {
                            sum[i][j] += regA[i] * regB[j];
                        }
                    }
                }

                // ---< 가져온 청크를 분해해서 다음 페이지 생성> --
                if(t < num_tiles-1){
                    current_page = !current_page; // 페이지 전환
                    if(tid < A_TILE_THREAD_COUNT) split_A(current_page, chunk_A);
                    if(tid < B_TILE_THREAD_COUNT) split_B(current_page, chunk_B);
                }

                // 동료들의 연산 종료를 기다린다.
                item.barrier(sycl::access::fence_space::local_space);
            }
            // 레지스터 -> 시스템 메모리로 정답을 돌려주면 끝.
            for (int i=0; i<BATCH_M; ++i)
            {
                for(int j=0; j<BATCH_N; ++j)
                {
                    d_R[(r*BATCH_M+i)*N + (c*BATCH_N+j)] = sum[i][j];
                }
            }
            });
        });
    // 결과를 기다리고, res를 return하면 끝... 이 아니라 사용한 메모리를 반드시 free시켜준다.
    q.wait();
    if(!has_system_usm)
    {
        // d_R -> res로 정답 행렬 복사
        q.memcpy(ptrR, d_R, M * N * sizeof(float)).wait();
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_R, q);
    }
    return res;
}
PYBIND11_MODULE(sycl_mat, m) {
    // 함수 자체를 노출시킴.
    m.def("matmul", &matmul, "SYCL based Matrix Multiplication");
}