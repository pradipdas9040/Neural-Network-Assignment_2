try:
    import cupy as np
    cupy = True
except ImportError:
    import numpy as np
    cupy = False

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def one_hot_encode(y, classes=None):
    y = np.array(y)
    if classes is None:
        classes = np.unique(y)

    n_classes = len(classes)

    out = np.zeros((len(y), n_classes))
    for i in range(n_classes):
        out[np.argwhere(y == classes[i]), i] = 1

    return out


def convert_to_binary(y: np.ndarray):
    return np.array(y.argmax(axis=1))


def accuracy(desired_op, pred_op):
    
    a = confusion_matrix(np.array(desired_op), np.array(pred_op))
    return np.trace(a) / np.sum(a)


def convert_to_cupy(x):
    if cupy:
        return np.array(x)
    else:
        return x


def convert_to_numpy(x):
    if cupy:
        return np.array(x)
    else:
        return 