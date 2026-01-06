MsDeep.py : 딥러닝으로 지뢰찾기의 각 미공개 칸의 확률을 예측하는 모델이 들어가는 곳.
layers.py : 넘파이로 직접 구현한 신경망의 각 계층 클래스가 들어가는 곳.
utils.py : 각종 유틸리티가 있음.
    - im2col(input_data, filter_h, filter_w, stride, pad)
        : 4차원 넘파이 배열과 필터 정보를 받아서 그것을 2차원 넘파이 배열로 펼쳐준다.
    - col2im()