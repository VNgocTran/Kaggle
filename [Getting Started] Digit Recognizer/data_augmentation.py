import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator

# open existed training set
with open("./data/X_train.p", "rb") as f:
    Xtrain = pickle.load(f)

# open existed training set
with open("./data/y_train.p", "rb") as f:
    ytrain = pickle.load(f)

# open existed test set
with open("./data/X_test.p", "rb") as f:
    X_test = pickle.load(f)

# train full set
# n_train = 42000

X_train = Xtrain
y_train = ytrain

X_train = np.reshape(X_train, (-1, 28,28,1))

def dataGen(X_train, y_train, nr_of_samples):
    new_x_data = []
    new_y_data = []
    r = 15
    for ite in range(nr_of_samples):
        datagen = ImageDataGenerator(
            rotation_range=r,
            width_shift_range=-0.1,
            height_shift_range=0,
            shear_range=0.1,
            zoom_range=0,
            horizontal_flip=False)

        for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=len(X_train), shuffle=False):
            print(x_batch.shape)
            new_x_data.append(np.reshape(x_batch, (-1, 28, 28, 1)))
            new_y_data.append(y_batch)
            break

    new_x_data = np.asanyarray(np.reshape(new_x_data, (-1, 28, 28, 1)))
    new_y_data = np.asanyarray(np.reshape(new_y_data, (-1,)))

    return new_x_data, new_y_data

x_new, y_new = dataGen(X_train, y_train, nr_of_samples=10)

with open("./data/X_train_new.p", "wb") as f:
    pickle.dump(x_new, f)

with open("./data/y_train_new.p", "wb") as f:
    pickle.dump(y_new, f)


