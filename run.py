import tensorflow as tf

from util.models import MyModel


def print_tensorflow():
    print("Tensorflow", tf.__version__)
    print("isBuild with CUDA:", tf.test.is_built_with_cuda())


def print_gpu_info():
    print("Number of GPU(s):", len(tf.config.list_physical_devices('GPU')))
    print("GPU(s):", tf.config.list_physical_devices('GPU'))


if __name__ == "__main__":

    print_tensorflow()
    print_gpu_info()

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = MyModel()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
