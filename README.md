# NaturalLanguage

## TensorFlow
2015년 구글에서 발표한 머신러닝 라이브러리

### tf.keras.layers.Dense (20p)
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


### tf.keras.layers.Dropout (22p)
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

### tf.keras.layers.Conv1D (25p)
| |합성곱의 방향|출력값|
|------|---|---|
|Conv1D|한 방향(가로)|1-D Array(vector)|
|Conv2D|두 방향(가로, 세로)|2-D Array(matrix)|
|Conv3D|세 방향(가로, 세로, 높이)|3-D Array(tensor)|

자연어 처라 분야에서 사용하는 합성곱의 경우 각 단어(혹은 문자) 벡터의 차원 전체에 대해 필터를 적용시키기 위해 주로 Conv1D를 사용한다.

### tf.keras.layers.Maxpool1D (28p)
피처 맵(feature map)의 크기를 줄이거나 주요한 특징을 뽑아내기 위해 합성곱 이후 적용되는 기법

- 맥스 풀링(max-pooling) : 최대값만 추출
- 평균 풀링(average-pooling) : 평균값 추출

Conv1D와 같이 Maxpool 또한 Maxpool(1D, 2D, 3D)중 자연어 처리에선 주로 1D를 사용한다.
