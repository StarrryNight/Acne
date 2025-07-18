from sklearn.utils import shuffle
import os
import numpy as np
from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler



dir_result = "result"
X = np.load(file = os.path.join(dir_result,"X.npy"))
y = np.load(file = os.path.join(dir_result,"y.npy"))
print(X)
print(y)
prop_train = 0.8

X, y = shuffle(X, y, random_state=0)



Ntrain = int(X.shape[0]*prop_train)
X_train, y_train, X_test, y_test = X[:Ntrain], y[:Ntrain], X[Ntrain:], y[Ntrain:]
print(X_train)
print(y_train)
# define the architecture of the network
model = Sequential()
model.add(Dense(32, input_dim=4096, activation="relu", kernel_initializer="uniform"))
model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
model.add(Dense(5, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def lr_scheduler(epoch, lr):
    decay_rate = 0.85
    decay_step = 50
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr


callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]
history = model.fit(X_train,
                    y_train,
                    validation_data = (X_test,y_test),
                    batch_size      = 64,
                    epochs       = 1000,
                    verbose         = 2,
                    callbacks=callbacks
                    )

fig = plt.figure(figsize=(20,5))
ax  = fig.add_subplot(1,2,1)
for key in ["val_loss","loss"]:
    ax.plot(history.history[key],label=key)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
plt.legend()
ax  = fig.add_subplot(1,2,2)
for key in ["val_accuracy","accuracy"]:
    ax.plot(history.history[key],label=key)
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
plt.legend()

plt.show()


model.save(os.path.join(dir_result,"classifier.h5"))
print("Saved model to disk")