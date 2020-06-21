#import keras.backend as K
import tensorflow.compat.v1.keras.backend as K
import tensorflow as tf

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

def get_gpu_session(ratio=None, interactive=False):
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    if ratio is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = ratio
    if interactive:
        sess = tf.InteractiveSession(config=config)
    else:
        sess = tf.compat.v1.Session(config=config)
    return sess


def set_gpu_usage(ratio=None):
    sess = get_gpu_session(ratio)
    K.set_session(sess)
