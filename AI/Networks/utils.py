import numpy as np;

# im2col 함수

def im2col(im, filter_h, filter_w, stride, pad):
    N, C, H, W = im.shape
    OH = (H+2*pad-filter_h)//stride + 1
    OW = (W+2*pad-filter_w)//stride + 1
    col = np.zeros((N*OH*OW, C*filter_h*filter_w))
    padded = np.pad(im, ((0,0), (0,0), (pad, pad), (pad, pad)))
    for i in range(N):
        for y in range(OH):
            for x in range(OW):
                col[i * OH * OW + y * OW + x,:] = padded[i, :, y*stride:y*stride+filter_h, x*stride:x*stride+filter_w].reshape((1, C*filter_h*filter_w))
    return col

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
