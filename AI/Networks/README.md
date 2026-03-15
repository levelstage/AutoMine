# Networks

TensorFlow, PyTorch 등의 프레임워크 없이 **NumPy만으로** 구현한 신경망 모듈입니다.

## 파일 구성

### `network.py` — DeepConvNet
각 레이어를 조립한 신경망 클래스. forward/backward 패스, 파라미터 저장/불러오기를 담당합니다.

### `layers.py` — 레이어 구현
역전파를 포함한 각 계층을 직접 구현했습니다.

| 클래스 | 설명 |
|---|---|
| `Conv2d` | 2D 합성곱 레이어 (im2col 활용) |
| `Affine` | 완전연결 레이어 |
| `ReLU` | ReLU 활성화 함수 |
| `Sigmoid` | Sigmoid 활성화 함수 |
| `BinaryCrossEntropy` | 이진 교차 엔트로피 손실 함수 |

### `optimizer.py` — Adam
Adam 옵티마이저. 편향 보정(bias correction)을 포함한 표준 구현입니다.

### `utils.py` — im2col / col2im
합성곱 연산을 행렬 곱으로 변환하는 핵심 유틸리티입니다.

- `im2col(input_data, filter_h, filter_w, stride, pad)` — 4D 입력 텐서를 2D 행렬로 펼침
- `col2im(col, input_shape, filter_h, filter_w, stride, pad)` — 역방향 변환 (역전파용)
