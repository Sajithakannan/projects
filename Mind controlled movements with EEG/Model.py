#MODEL:

import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Activation
import time
import matplotlib.pyplot as plt

ACTIONS = ["D:\\College\\Mini Project\\model_data\\data\\left", 
           "D:\\College\\Mini Project\\model_data\\data\\right", 
           "D:\\College\\Mini Project\\model_data\\data\\none"]
reshape = (-1, 16, 60)

def create_data(starting_dir):
    training_data = {}
    for action in ACTIONS:
        training_data[action] = []
        data_dir = os.path.join(starting_dir, action)
        for item in os.listdir(data_dir):
            data = np.load(os.path.join(data_dir, item))
            for item in data:
                training_data[action].append(item)

    min_length = min(len(training_data[action]) for action in ACTIONS)
    for action in ACTIONS:
        np.random.shuffle(training_data[action])
        training_data[action] = training_data[action][:min_length]

    combined_data = []
    for action in ACTIONS:
        for data in training_data[action]:
            if action == ACTIONS[0]:
                combined_data.append([data, [1, 0, 0]])
            elif action == ACTIONS[1]:
                combined_data.append([data, [0, 0, 1]])
            elif action == ACTIONS[2]:
                combined_data.append([data, [0, 1, 0]])

    np.random.shuffle(combined_data)
    return combined_data

print("Creating training data")
train_data = create_data("D:\\College\\Mini Project\\model_data\\data")
train_X = np.array([x[0] for x in train_data]).reshape(reshape)
train_y = np.array([x[1] for x in train_data])

print("Creating testing data")
test_data = create_data("D:\\College\\Mini Project\\model_data\\validation_data")
test_X = np.array([x[0] for x in test_data]).reshape(reshape)
test_y = np.array([x[1] for x in test_data])

model = Sequential()
model.add(Conv1D(64, (3), input_shape=train_X.shape[1:]))
model.add(Activation('relu'))
model.add(Conv1D(128, (2)))
model.add(Activation('relu'))
model.add(Conv1D(128, (2)))
model.add(Activation('relu'))
model.add(Conv1D(64, (2)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(64, (2)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 10
batch_size = 32
for epoch in range(epochs):
    model.fit(train_X, train_y, batch_size=batch_size, epochs=1, validation_data=(test_X, test_y))
    score = model.evaluate(test_X, test_y, batch_size=batch_size)
    #sajimodel = f"new_models/{round(score[1]*100,2)}-acc-64x3-batch-norm-{epoch}epoch-{int(time.time())}-loss-{round(score[0],2)}.model"
    sajimodel = f"new_models/{round(score[1]*100,2)}-acc-64x3-batch-norm-{epoch}epoch-{int(time.time())}-loss-{round(score[0],2)}.h5"
    model.save(sajimodel)
    print("Model saved:", sajimodel)