import tensorflow as tf

def print_tensorflow():
    """
    Output Tensorflow related build
    """
    print("Tensorflow", tf.__version__)
    print("isBuild with CUDA:", tf.test.is_built_with_cuda())


def print_gpu_info():
    """
    Output GPU(s) info that is compatible to CUDA on the machine
    """
    print("Number of GPU(s):", len(tf.config.list_physical_devices('GPU')))
    print("GPU(s):", tf.config.list_physical_devices('GPU'))
