import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import initializers
import tensorflow as tf

#y = 3*x1 + 5*x2 + 10

tf.random.set_seed(42)
def gen_sequential_model():
    model = Sequential([
            Input(2, name='input_layer'),
            Dense(16, activation='sigmoid', name='hidden_layer1', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)), #뉴런의 weight값을, 초기 랜덤값을 고정.
            Dense(1, activation='relu', name='output_layer', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))
            ])
            
    model.summary()
    #print(model.layers[0].get_weights())
    #print(model.layers[1].get_weights())
    
    model.compile(optimizer='sgd', loss='mse') #optimizer는 ~gradient descent. mse는 mean of square error. predefine 되어있는 것을 사용.
    #MultiClassification같은건 cross entropy 같은걸 씀. 찾아보면 다른 optimizer나 loss function이 많이 있음.

    return model


#y = w1*x1 + w2*x2 + b

def gen_linear_regression_dateset(numofsamples=500, w1=3, w2=5, b=10):

    np.random.seed(42)
    X = np.random.rand(numofsamples, 2)
    print(X)
    print(X.shape)
    
    coef = np.array([w1, w2])
    bias = b
    
    print(coef)
    print(coef.shape)
    
    y = np.matmul(X, coef.transpose()) + bias  #transpose()가 가로세로를 바꿔줌.  vector 계산을 함으로써 for문 사용하지 않음.
    
    #X=(numofsamples, 2), coef.transpose() = (2,1) 차원.  따라서 y는 (numofsamples, 1)차원.
    
    print(y)
    print(y.shape)
    
    return X, y


def plot_loss_curve(history):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()   

def predict_new_sample(model, x, w1=3, w2=5, b=10):
    
    x = x.reshape(1,2) #1차원의 x를 1x2차원으로 수정.
    y_pred = model.predict(x)[0][0]
    
    y_actual = w1*x[0][0] + w2*x[0][1] + b
    
    print("y actual value = ", y_actual)
    print("y predicted value = ", y_pred)
    

model = gen_sequential_model()
X, y = gen_linear_regression_dateset(numofsamples=1000)
history = model.fit(X, y, epochs=200, verbose=2, validation_split=0.3)  #verbose는 실제 트레이닝하는 과정 각 단계에서의 loss값을 표현.
                                                #validation_split은 나누는것. 트레이닝에 70%, test(validate)하는데 30%를 쓰겠다는 말.
plot_loss_curve(history)  #그래프 표현.
print("train loss=", history.history['loss'][-1])  #loss 출력 (아랫줄은 validation에 대한 loss.)
print("test loss=", history.history['val_loss'][-1])

predict_new_sample(model, np.array([0.6, 0.3]))    #0과1사이값으로 학습해서 넣는 값도 이를 권장.
    
    

