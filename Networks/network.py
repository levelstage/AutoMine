from collections import OrderedDict
from . import layers
import pickle

class DeepConvNet:
    def __init__(self, input_dim=(10, 10, 10),
                 hidden_size=256, output_size=100):
        # 입력받은 형상을 쓰기 좋게 분할해준다
        C, H, W = input_dim
        
        # conv2의 출력 개수를 미리 계산해둔다(Affine과 연결하기 위해)
        pool_output_size = 64 * H * W
        
        # layer들을 저장할 dictionary
        self.layers = OrderedDict()

        # 1층: 게임판 모양의 기초적인 패턴을 추출해줄 Conv1 세팅
        self.layers['Conv1'] = layers.Conv2d(C, 32, 3, 1, 1)
        self.layers['Relu1'] = layers.ReLU()

        # 2층: 패턴들의 상호작용을 분석해줄 Conv2 세팅
        self.layers['Conv2'] = layers.Conv2d(32, 32, 3, 1, 1)
        self.layers['Relu2'] = layers.ReLU()

        # 3,4 층: 더 깊은 패턴 분석을 위한 층 추가.
        self.layers['Conv3'] = layers.Conv2d(32, 64, 3, 1, 1)
        self.layers['Relu3'] = layers.ReLU()

        self.layers['Conv4'] = layers.Conv2d(64, 64, 3, 1, 1)
        self.layers['Relu4'] = layers.ReLU()


        # 은닉층: 완전연결을 통해 추론을 진행할 Affine 세팅
        self.layers['Affine1'] = layers.Affine(pool_output_size, hidden_size)
        self.layers['Relu5'] = layers.ReLU()

        # 출력을 위해 마지막으로 합성곱을 진행할 Affine 세팅
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
        mask = x[:,0].reshape(x.shape[0], -1)
        return self.last_layer.forward(y, t, mask)
    
    def gradient(self, x, t):
        # 순전파 (미분값 저장을 위한)
        self.loss(x, t)

        # 역전파
        dout = self.last_layer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 미분값 매핑하여 저장
        grads = {}
        i = 1
        for layer in self.layers.values():
            if hasattr(layer, 'W'):
                grads['W'+str(i)] = layer.dW
                grads['b'+str(i)] = layer.db
                i+=1
        
        return grads
    
    def save_params(self, file_name="my_model.pkl"):
        params = {}
        i = 1
        # 현재 레이어들의 W, b를 싹 긁어모아서 딕셔너리에 담음
        for layer in self.layers.values():
            if hasattr(layer, 'W') and layer.W is not None:
                params['W' + str(i)] = layer.W
                params['b' + str(i)] = layer.b
                i += 1
        
        # 파일로 저장 (wb: write binary)
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        print(f"학습된 가중치를 저장했습니다: {file_name}")

    def load_params(self, file_name="my_model.pkl"):
        try:
            # 파일 읽기 (rb: read binary)
            with open(file_name, 'rb') as f:
                params = pickle.load(f)
            
            # 읽어온 가중치를 레이어에 다시 끼워넣기
            i = 1
            for layer in self.layers.values():
                if hasattr(layer, 'W') and layer.W is not None:
                    layer.W = params['W' + str(i)]
                    layer.b = params['b' + str(i)]
                    i += 1
            print(f"가중치 로드 성공! 학습 없이 바로 사용 가능합니다: {file_name}")
            return True
        except FileNotFoundError:
            print(f"저장된 파일({file_name})이 없습니다. 새로 학습해야 합니다.")
            return False
