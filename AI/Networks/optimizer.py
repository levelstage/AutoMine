import numpy as np

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, net, grad):
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in grad.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        # 초기값이 0이기 때문에 정상적인 학습을 위해 값을 보정해줘야함.
        self.iter += 1
        j=1
        for layer in net.layers.values():
            if hasattr(layer, 'W'):
                # 가중치 갱신
                self.m['W'+str(j)] = self.beta1 * self.m['W'+str(j)] + (1-self.beta1) * grad['W'+str(j)]
                self.v['W'+str(j)] = self.beta2 * self.v['W'+str(j)] + (1-self.beta2) * (grad['W'+str(j)]**2)
                m_corrected = self.m['W'+str(j)] / (1 - self.beta1 ** self.iter)
                v_corrected = self.v['W'+str(j)] / (1 - self.beta2 ** self.iter)
                layer.W -= self.lr * m_corrected / (np.sqrt(v_corrected) + 1e-8)
                # 바이어스 갱신
                self.m['b'+str(j)] = self.beta1 * self.m['b'+str(j)] + (1-self.beta1) * grad['b'+str(j)]
                self.v['b'+str(j)] = self.beta2 * self.v['b'+str(j)] + (1-self.beta2) * (grad['b'+str(j)]**2)
                m_corrected = self.m['b'+str(j)] / (1 - self.beta1 ** self.iter)
                v_corrected = self.v['b'+str(j)] / (1 - self.beta2 ** self.iter)
                layer.b -= self.lr * m_corrected / (np.sqrt(v_corrected) + 1e-8)
                j+=1