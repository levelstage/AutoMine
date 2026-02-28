import numpy as np;

# im2col 함수
def im2col(im, filter_h, filter_w, stride, pad):
    # N, C, H, W
    N, C, H, W = im.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    # 패딩 처리
    img = np.pad(im, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    
    # [N, C, H, W] 이미지에서 [N, C, out_h, out_w, filter_h, filter_w] 6차원 뷰를 생성
    # 메모리 복사 없이 stride(보폭)만 조작해서 겹치는 부분을 표현함 (매우 빠름)
    col = np.lib.stride_tricks.as_strided(
        img,
        shape=(N, C, out_h, out_w, filter_h, filter_w),
        strides=(img.strides[0], img.strides[1], img.strides[2]*stride, img.strides[3]*stride, img.strides[2], img.strides[3])
    )

    # 행렬 곱셈을 위해 (N*OH*OW, C*FH*FW) 형상으로 정렬 후 반환
    res = col.transpose(0, 2, 3, 1, 4, 5).reshape(N*out_h*out_w, -1)
    return np.ascontiguousarray(res, dtype=np.float32)

#  col2im 함수

def col2im(col, input_shape, filter_h, filter_w, stride, pad):
    N, C, H, W = input_shape
    OH = (H+2*pad-filter_h)//stride + 1
    OW = (W+2*pad-filter_w)//stride + 1
    # N, OH, OW, C, filter_h, filter_w
    padded = np.zeros((N, C, H + 2*pad, W + 2*pad))
    for i in range(N):
        for y in range(OH):
            for x in range(OW):
                padded[i, :, y*stride:y*stride+filter_h, x*stride:x*stride+filter_w] += col[i * OH * OW + y * OW + x, :].reshape(C, filter_h, filter_w)
    if pad == 0:
        return padded
    return padded[:, :, pad:-pad, pad:-pad]
