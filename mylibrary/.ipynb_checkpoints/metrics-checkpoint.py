import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.metrics import binary_accuracy

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


@tf.function
def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    F1 score: https://en.wikipedia.org/wiki/F1_score
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

        ##clip:指定范围数变边界值
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return float(0)

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def invasion_acc(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return binary_accuracy(binary_truth, binary_pred)


def invasion_precision(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return precision(binary_truth, binary_pred)


def invasion_recall(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return recall(binary_truth, binary_pred)


def invasion_fmeasure(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return fmeasure(binary_truth, binary_pred)


def ia_acc(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return binary_accuracy(binary_truth, binary_pred)


def ia_precision(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return precision(binary_truth, binary_pred)


def ia_recall(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return recall(binary_truth, binary_pred)


def ia_fmeasure(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return fmeasure(binary_truth, binary_pred)
