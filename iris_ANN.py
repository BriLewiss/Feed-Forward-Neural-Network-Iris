from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


# Function to initialize weights
def init_weights(shape):
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

# Function to hot encode labels
def one_hot_encode(labels, num_classes):
    # Create an array of zeros with shape (len(labels), num_classes)
    one_hot = np.zeros((len(labels), num_classes))
    # Set the appropriate element to one
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


iris = datasets.load_iris()
X,y = iris.data, iris.target
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.50)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


# hot encode the data so it is in the correct format for the neural network
num_classes = 3  # As there are three classes in the Iris dataset
y_train = one_hot_encode(y_train, num_classes)
y_test = one_hot_encode(y_test, num_classes)
y_val = one_hot_encode(y_val, num_classes)


# Create the model
model = keras.Sequential()
model.add(keras.layers.Dense(6, activation='relu', use_bias=True, bias_initializer="zeros", input_shape=(4,))) # 2 hidden nodes
model.add(keras.layers.Dense(3, activation='softmax')) # 3 output nodes
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train,
          batch_size=12, # number of samples per gradient update
          epochs=200, # number of iterations
          validation_data=(X_val, y_val))

# Evaluate the model on the test data
_, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)