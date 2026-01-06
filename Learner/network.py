import numpy as np
from collections import OrderedDict
import layers

class DeepConvNet:
    def __init__(self, input_dim=(10, 10, 10), 
                 conv1_p={'FN': 32, 'FS': 3, 'stride': 1, 'pad': 1},
                 conv2_p={'FN': 64, 'FS': 3, 'stride': 1, 'pad': 1},
                 hidden_size=256, output_size=100):
        # 입력받은 형상을 쓰기 좋게 분할해준다
        C, H, W = input_dim
        
        # conv2의 출력 개수를 미리 계산해둔다(Affine과 연결하기 위해)
        pool_output_size = conv2_p['FN'] * H * W
        
        # layer들을 저장할 dictionary
        self.layers = OrderedDict()

        # 1층: 게임판 모양의 기초적인 패턴을 추출해줄 Conv1 세팅
        self.layers['Conv1'] = layers.Conv2d(C, conv1_p['FN'], conv1_p['FS'], conv1_p['stride'], conv1_p['pad'])
        self.layers['Relu1'] = layers.ReLU()

        # 2층: 패턴들의 상호작용을 분석해줄 Conv2 세팅
        self.layers['Conv2'] = layers.Conv2d(C, conv2_p['FN'], conv2_p['FS'], conv2_p['stride'], conv2_p['pad'])
        self.layers['Relu2'] = layers.ReLU()

        # 3층: 은닉층. 완전연결을 통해 추론을 진행할 Affine 세팅
        self.layers['Affine1'] = layers.Affine(pool_output_size, hidden_size)
        self.layers['Relu3'] = layers.ReLU()

        # 4층: 출력을 위해 마지막으로 합성곱을 진행할 Affine 세팅
        self.layers['Affine2'] = layers.Affine(hidden_size, output_size)

        # 출력&오차: 시그모이드와 CrossBinaryEntropy로 출력 및 오차를 담당함
        self.layers['Sigmoid1'] = layers.Sigmoid()
        self.last_layer = layers.BinaryCrossEntropy()

        def predict(self, x):
            for layer in self.layers.values():
                x = layer.forward(x)
            return x
        
        def loss(self, x, t):
            y = self.predict(x)
            return self.last_layer.forward(y, t)
        
        def gradient(self, x, t):
            # 순전파 (미분값 저장을 위한)
            self.loss(x, t)

            # 역전파        
            self.last_layer.backward()
            layers = list(self.layers.values())
            layers.reverse()
            for layer in layers:
                dout = layer.backward(dout)
            
            # 미분값 매핑하여 저장
            grads = {}
            i = 1
            for layer in self.layers.values():
                grads['W'+str(i)] = layer.dW
                grads['b'+str(i)] = layer.db
                i+=1
            
            return grads
