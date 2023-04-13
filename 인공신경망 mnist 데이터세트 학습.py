import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# MNIST 데이터셋 불러오기
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 인공신경망 모델 생성
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 이미지를 1차원 벡터로 변환
    Dense(128, activation='relu'),  # 128개의 노드를 가진 은닉층
    Dense(10, activation='softmax')  # 10개의 노드를 가진 출력층 (각 클래스에 대한 확률)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(x_train, y_train, epochs=5)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print('테스트 정확도:', test_acc)
