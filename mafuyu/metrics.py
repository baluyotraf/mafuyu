import matplotlib.pyplot as plt
import numpy as np
import itertools
from .utils import print_distribution
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score
try:
    import enum
except ImportError:
    import enum43 as enum


class EpochMetrics:

    def __init__(self):
        self._iteration_data = []
        self._epoch_data = []

    def add_iteration_data(self, value, batch_size=1):
        iter_data = (value * batch_size, batch_size)
        self._iteration_data.append(iter_data)

    def end_epoch(self):
        values, sizes = tuple(zip(*self._iteration_data))
        epoch_data = np.sum(values) / np.sum(sizes)
        self._epoch_data.append(epoch_data)
        self._iteration_data.clear()

    def clear(self):
        self._iteration_data.clear()
        self._epoch_data.clear()

    @property
    def data(self):
        return self._epoch_data


def plot_counts(iterable, plot_size=(10, 5)):
    count = Counter(iterable)
    item, num = zip(*count.items())
    plt.figure(figsize=plot_size)
    rs = plt.barh(item, num)
    for r, n in zip(rs, num):
        x = 1.005 * r.get_width()
        y = r.get_y() + r.get_height() / 2
        plt.text(x, y, str(n), ha='left', va='center')


def print_regression_metrics(true, pred, rows=3, plot_size=(10, 2)):
    true, pred = np.asarray(true), np.asarray(pred)

    mae = np.abs(true - pred)
    mse = mae ** 2

    print_distribution(mae, 'MAE', rows, plot_size)
    print_distribution(mse, 'MSE', rows, plot_size)


class CMType(enum.Enum):
    SUPPORT = 'support'
    RECALL = 'recall'
    PRECISION = 'precision'


def plot_confusion_matrix(cm, classes,
                          cm_type=CMType.SUPPORT,
                          title='Confusion matrix',
                          plot_size=(5, 5),
                          cmap=plt.cm.Blues):
    normalize = True
    if cm_type == CMType.RECALL:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif cm_type == CMType.PRECISION:
        cm = cm.astype(float) / cm.sum(axis=0)
    elif cm_type == CMType.SUPPORT:
        normalize = False
    else:
        raise ValueError('cm_type must be a {} instance'.format(CMType.__class__.name))

    plt.figure(figsize=plot_size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def print_classification_metrics(true, pred, classes, plot_size=(5, 5)):
    acc = accuracy_score(true, pred)
    print('ACC: ', acc)

    cm = confusion_matrix(true, pred)
    plot_confusion_matrix(cm, classes, CMType.SUPPORT,
                          'Support Confusion Matrix', plot_size)
    plot_confusion_matrix(cm, classes, CMType.RECALL,
                          'Recall Confusion Matrix', plot_size)
    plot_confusion_matrix(cm, classes, CMType.PRECISION,
                          'Precision Confusion Matrix', plot_size)

