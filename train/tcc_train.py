from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Dropout, Input, Concatenate
from tensorflow.keras.regularizers import l2
from point_to_picture import keypoints_to_image
import numpy as np
class TCcnn(keras.Model):
  def __init__(self, input_size1, input_size2):
    super().__init__()
    print('tcnn input: ', input_size1, input_size2)

    n_features = input_size1[-1]
    # (n_freqs, n_features)
    self.cnn1d = Sequential([
       Input(shape=input_size1),
       Conv1D(filters=6 * n_features, kernel_size=6, activation='relu'),
       MaxPooling1D(pool_size=4),
       Conv1D(filters=16 * n_features, kernel_size=5, activation='relu'),
       MaxPooling1D(pool_size=3),
       Flatten()
    ])

    self.cnn2d = Sequential([
       Input(shape=input_size2),
       Conv2D(filters=6 * n_features, kernel_size=5, activation='relu'),
       MaxPooling2D(pool_size=2),
       Conv2D(filters=16 * n_features, kernel_size=5, activation='relu'),
       MaxPooling2D(pool_size=2),
       Flatten()
    ])

    self.dropout = Dropout(0.5)
    self.fc1 = Dense(84, activation='relu')

    # SVM
    self.fc2 = Dense(1, kernel_regularizer=l2(0.01))
    # Softmax()

  def build(self, input_size1, input_size2):
    # self.cnn1d.build(input_size1)
    # self.cnn2d.build(input_size2)
    input_shape1 = self.cnn1d.compute_output_shape((None, *input_size1))
    input_shape2 = self.cnn2d.compute_output_shape((None, *input_size2))
    self.fc1.build((None, input_shape1[1]+input_shape2[1]))
    input_shape = self.fc1.compute_output_shape((None, input_shape1[1]+input_shape2[1]))
    self.dropout.build(input_shape)
    input_shape = self.dropout.compute_output_shape(input_shape)
    self.fc2.build(input_shape)
    # self.dense2.build(input_shape)
    # print(input_shape1, input_shape2)

  def call(self, inputs):
    fft, sfft = inputs
    x1 = self.cnn1d(fft)
    x2 = self.cnn2d(sfft)
    x = Concatenate(axis=-1)([x1, x2])
    x = self.dropout(x)
    x = self.fc1(x)
    pred = self.fc2(x)
    return keras.activations.sigmoid(pred)

def run_tccnn(x_train, y_train, x_test, y_test, epochs=500):
  # input size確認，
  input_size1 = x_train[0].shape[1:]
  input_size2 = x_train[1].shape[1:]
  model = TCcnn(input_size1, input_size2)
  model.build(input_size1, input_size2)
  model.summary()
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # hinge
  # fit network
  model.fit(x_train, y_train, epochs=epochs, batch_size=8)
  # evaluate model
  _, accuracy = model.evaluate(x_test, y_test, batch_size=10, verbose=0)
  y_pred = model.predict(x_test)
  # print(list(zip(y_pred.tolist(), y_test)))
  print("Accuracy of Model::", accuracy)
  return model, accuracy, list(zip(y_pred.tolist(), y_test))

keypoints_data = np.load("X.npy")

# 將關鍵點轉換成圖片並儲存
# for i, keypoints in enumerate(keypoints_data):
#     image = keypoints_to_image(keypoints)

