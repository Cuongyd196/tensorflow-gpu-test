import tensorflow as tf

from util.models import MyModel

# Output Tensorflow related build
def print_tensorflow():
    print("Tensorflow", tf.__version__)
    print("isBuild with CUDA:", tf.test.is_built_with_cuda())


# Output GPU(s) info that is compatible to CUDA on the machine
def print_gpu_info():
    print("Number of GPU(s):", len(tf.config.list_physical_devices('GPU')))
    print("GPU(s):", tf.config.list_physical_devices('GPU'))


if __name__ == "__main__":

    # Output info about tensorflow and gpu(s)
    print_tensorflow()
    print_gpu_info()

    # Retrieve the dataset
    mnist = tf.keras.datasets.mnist

    # Split dataset into train and test datasets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Create a model by grouping layers into an object
    model = MyModel()

    # Configures the model for training
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(x_train, y_train, epochs=5)

    # Evaluate the model
    model.evaluate(x_test, y_test)
