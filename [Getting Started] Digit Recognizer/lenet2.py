import numpy as np
import tensorflow as tf
import pickle

from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from tensorflow.contrib.layers import flatten

# open existed training set
with open("./data/X_train.p", "rb") as f:
    Xtrain = pickle.load(f)

# open existed training set
with open("./data/y_train.p", "rb") as f:
    ytrain = pickle.load(f)
    
X_train = Xtrain
y_train = ytrain

X_train = np.reshape(X_train, (-1, 28,28,1))

# open existed training set
with open("./data/X_train_new.p", "rb") as f:
    X_train_new = pickle.load(f)

# open existed training set
with open("./data/y_train_new.p", "rb") as f:
    y_train_new = pickle.load(f)
	
X_new = np.asarray(np.ndarray.tolist(X_train) + np.ndarray.tolist(X_train_new))
print(X_new.shape)
print(type(X_new))

y_new = np.asarray(np.ndarray.tolist(y_train) + np.ndarray.tolist(y_train_new))


X_new = np.reshape(X_new, (-1, 28,28,1))
X_new = X_new/255

# Building CNN Model
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 28x28x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(1, 1, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)
    
    #-----------------------------------------------------------------
    # Layer 1-1: Convolutional. Input = 28x28x6. Output = 24x24x16.
    conv1_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 16), mean = mu, stddev = sigma))
    conv1_1_b = tf.Variable(tf.zeros(16))
    conv1_1 = tf.nn.conv2d(x, conv1_1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_1_b

    # Activation.
    conv1_1 = tf.nn.relu(conv1_1)

    # Pooling. Input = 26x26x16. Output = 13x13x6.
    conv1_1 = tf.nn.max_pool(conv1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #-----------------------------------------------------------------
    
    
    # Layer 2: Convolutional. Output = 11x11x40.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 40), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(40))
    conv2 = tf.nn.conv2d(conv1_1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 8x8x22. Output = 5x5x40.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x40. Output = 1000.
    fc0 = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 1000. Output = 100.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1000, 100), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(100))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 100. Output = 40.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(100, 40), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(40))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 40. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(40, 10), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
	
EPOCHS = 30
BATCH_SIZE = 100

x = tf.placeholder(tf.float32, (None, 28, 28, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
	
model_folder = "./model3"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_new)
    
    print("Start training ...")
    print()
    for i in range(EPOCHS):
        X_new, y_new = shuffle(X_new, y_new)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_new[offset:end], y_new[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
#         validation_accuracy = evaluate(X_validation, y_validation)
        train_accuracy = evaluate(X_new, y_new)
        print("EPOCH {} ...".format(i+1))
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        print()
        
    saver.save(sess, model_folder + "/trained_model.ckpt")
    print("Model saved")
	
	
	
	
	
	
	
with tf.Session() as sess:
    # Load the saved model
    saver.restore(sess, tf.train.latest_checkpoint(model_folder))

    # Training set accuracy
    training_accuracy = evaluate(X_train, y_train)
    print("Training Accuracy = {:.3f}".format(training_accuracy))

# open existed test set
with open("./data/X_test.p", "rb") as f:
    X_test = pickle.load(f)
X_test = np.reshape(X_test, (-1, 28,28,1))

with tf.Session() as sess:
    # Load trained model
    saver.restore(sess, tf.train.latest_checkpoint("./model3"))
    test_logits = sess.run(logits, feed_dict={x: X_test})
    prediction = tf.argmax(test_logits, 1)
    
    p = sess.run(prediction)

print("ImageId,Label")
for ite, row in enumerate(p):
    print("{},{} ".format(ite+1, row))
