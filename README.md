# NaturalLanguage

## 1. 모델구축

## tf.keras.layers.Dense (20p)
```python
# 1. 객체 생성 후 다시 호출하면서 입력값 설정
dense = tf.keras.layers.Dense( ... )
output = dense(input)

# 2. 객체 생성 시 입력값 설정
output = tf.keras.layers.Dense( ... )(input)
```

신경망의 가장 기본적인 형태
- units
    - 출력 값의 크기, integer 혹은 Long 형태
- activation = None
    - 활성화 함수
- use_bias = True
    - 편향(b)을 사용할지 여부, Boolean 값 형태
- kernel_initializer = 'glorot_uniform'
    - 가중치(W) 초기화 함수
- bias_initializer = 'zeros'
    - 편향 초기화 함수
- kernel_regularizer = None
    - 가중치 정규화 방법
- bias_regularizer = None
    - 편향 정규화 방법
- activity_regularizer = None
    - 출력 값 정규화 방법
- kernel_constraint = None
    - Optimizer에 의해 업데이트된 이후 가중치에 적용되은 부가적인 제약 함수 (ex. norm constraint, value constraint)
- bias__constraint = None
    - Optimizer에 의해 업데이트된 이후 편향에 적용되은 부가적인 제약 함수 (ex. norm constraint, value constraint)


## tf.keras.layers.Dropout (22p)
신경망 모델의 과적합(Overfitting) 문제를 해결하는 정규화(Regularization)의 대표적 방법
```python
# 1. 객체 생성 후 다시 호출하면서 입력값 설정
dropout = tf.keras.layers.Dropout( ... )
output = dropout(input)

# 2. 객체 생성 시 입력값 설정
output = tf.keras.layers.Dropout( ... )(input)
```
- rate
    - 드롭아웃을 적용할 확률을 지정 (0 ~ 1 사이의 값)
- noise_shape = None
    - 정수형의 1D-tensor 값을 받는다. 여기서 받은 값은 shape을 뜻한다.
- seed = None
    - 임의의 선택을 위한 시드 값

## tf.keras.layers.Conv1D (25p)
| |합성곱의 방향|출력값|
|------|---|---|
|Conv1D|한 방향(가로)|1-D Array(vector)|
|Conv2D|두 방향(가로, 세로)|2-D Array(matrix)|
|Conv3D|세 방향(가로, 세로, 높이)|3-D Array(tensor)|

자연어 처라 분야에서 사용하는 합성곱의 경우 각 단어(혹은 문자) 벡터의 차원 전체에 대해 필터를 적용시키기 위해 주로 Conv1D를 사용한다.

## tf.keras.layers.Maxpool1D (28p)
피처 맵(feature map)의 크기를 줄이거나 주요한 특징을 뽑아내기 위해 합성곱 이후 적용되는 기법

- 맥스 풀링(max-pooling) : 최대값만 추출
- 평균 풀링(average-pooling) : 평균값 추출

Conv1D와 같이 Maxpool 또한 Maxpool(1D, 2D, 3D)중 자연어 처리에선 주로 1D를 사용한다.

## Sequential API (tf.keras.Sequential)
keras를 활용하여 모델을 구축할 수 있는 가장 간단한 형태의 API
```python
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation = 'rulu'))
model.add(layers.Dense(10, activation = 'softmax'))
```

## Functional API (32p)
- Sequential API 사용이 어려운 경우 사용
    - 다중 입력값 모델 (Multi-input models)
    - 다중 출력값 모델 (Multi-output models)
    - 공유 층을 활용하는 모델 (Models with shared layers)
    - 데이터 흐름이 순차적이지 않은 모델 (Models with non-sequential data flows)

```python
inputs = tf.keras.input(shape=(32,))
x = layers.Dense(64, activation = 'rulu')(inputs)
x = layers.Dense(64, activation = 'rulu')(x)
predictions = layers.Dense(10, activation = 'sofrmax')(x)
```

## Custom Layer (33p)
새로운 연산을 하는 레이어 혹은 편의를 위해 여러 레이어를 하나로 묶은 레이어를 구현해야 하는 경우 사용자 정의 층(Custom Layer)을 만들어 사용
```python
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(CustomLayer(64, 64, 10))
```

## Subclassing (Custom Model)
tf.keras.Model을 상속받고 모델 내부 연산자들을 직접 구현
```python
class MyModel(tf.keras.Model):

    def __init__(self, hidden_dimension, hidden_dimension2, output_dimension):
        super(MyModel, self).__init__(name = 'my model')
        self.dense_layer1 = layers.Dense(hidden_dimension, activation = 'relu')
        self.dense_layer2 = layers.Dense(hidden_dimension2, activation = 'relu')
        self.dense_layer3 = layers.Dense(output_dimension, activation = 'softmax')

    def cal(self, inputs):
        x = self.dense_layer1(input)
        x = self.dense_layer2(x)

        return self.dense_layer3(x)
```

## 2. 모델 학습