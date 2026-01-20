network.py : 각 layer들을 조립한 신경망 클래스가 들어가는 곳.
layers.py : 넘파이로 직접 구현한 신경망의 각 계층 클래스가 들어가는 곳.
utils.py : 각종 유틸리티가 있음.
    - im2col(input_data, filter_h, filter_w, stride, pad)
        : 4차원 넘파이 배열과 필터 정보를 받아서 그것을 2차원 넘파이 배열로 펼쳐준다.
    - col2im(col, input_shape, filter_h, filter_w, stride, pad)
        : 2차원으로 펼쳐진 4차원 넘파이 배열을 원래 형상으로 되돌린다.