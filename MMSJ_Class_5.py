#-*-coding:utf-8-*-
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import keras
import itertools

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_score, confusion_matrix, accuracy_score, classification_report, cohen_kappa_score,roc_curve, auc
from itertools import cycle
from scipy import interp
from tqdm import tqdm
from glob import glob


def pad(data, sw):
    """
    padding and clipping data with special length
    :param data: original ecg sequence data
    :param sw: length of sliding window
    :return:
    """
    data = np.squeeze(data)
    data_new = np.zeros(sw, dtype=np.float32)
    data_new[sw / 2 - data.shape[0] / 2: sw / 2 - data.shape[0] / 2 + data.shape[0]] = data
    return data_new


def load_data(data_dir="./MIT-BIH/", data_num_threshold=7000, sw=3000):
    """
    loading all ecg data
    :param data_dir: ecg data directory
    :param data_num_threshold: if the number of data with specific class great than threshold, then add them to dataset
    :param sw: the length of sliding window
    :return: loaded data_x and data_y with label mapping rules
    """
    data_files = glob(data_dir + "*.dat")
    data_x = []
    data_y = []
    for data_file in tqdm(data_files):
        # reading data id
        data_id = data_file[data_file.rfind("/") + 1: data_file.rfind(".dat")]
        # load original data
        record = wfdb.rdrecord(data_dir + data_id, sampfrom=0, sampto=650000, channels=[0])
        # load data label
        ann = wfdb.rdann(data_dir + data_id, 'atr', sampfrom=0, sampto=650000, return_label_elements=['label_store'])

        rr_indexes = ann.sample  # obtain the index of RR peak
        labels = ann.label_store  # obtain all label of an ECG sample
        ecg_data = record.p_signal # obtain original data signal

        # sample data from whole ECG sequence and ensure the center of clipped data equals to the index of RR peak
        for i in range(rr_indexes.shape[0] - 2):
            data_range_min = rr_indexes[i]
            data_range_max = rr_indexes[i + 2]
            data_label = labels[i + 1]
            data_x.append(pad(ecg_data[data_range_min: data_range_max], sw))
            data_y.append(data_label)

    # select types of training data which the number of sample is greater than threshold
    selected_classes = []
    for label in set(data_y):
        if np.sum(data_y == label) > data_num_threshold:
            selected_classes.append(label)

    # print selected types
    print "selected classes: {}".format(selected_classes)

    # balance dataset
    selected_indexes = []
    for i in range(len(data_x)):
        if data_y[i] in selected_classes:
            if data_y[i] == 1:
                if np.random.randint(20) < 2:
                    selected_indexes.append(i)
            else:
                selected_indexes.append(i)

    # generate label mapping as a map set
    label_mapping = {}
    for i, clazz in enumerate(selected_classes):
        label_mapping[clazz] = i

    print "label mapping: {}".format(label_mapping)
    label_mapping_new = {v: k for k, v in label_mapping.items()}
    print "label mapping New: {}".format(label_mapping_new)

    selected_data_x = []
    selected_data_y = []
    for i in selected_indexes:
        selected_data_x.append(data_x[i])
        selected_data_y.append(label_mapping[data_y[i]])

    print "successfully load {} data_x, {} data_y".format(len(selected_data_x), len(selected_data_y))

    selected_data_x = np.array(selected_data_x, dtype=np.float32)
    selected_data_y = np.array(selected_data_y, dtype=np.float32)
    return selected_data_x, selected_data_y, label_mapping


class LossHistory(keras.callbacks.Callback):
    """
    define a callback function which will be called by deep model training function
    """

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train_accuracy')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train_loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val_accuracy')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val_loss')

        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')

        plt.legend(loc='upper right')
        plt.savefig("image_acc-loss.tif", dpi=300)
        plt.show()


if __name__ == '__main__':
    # randomly selecting 20% of data as test data
    data_x, data_y, label_map = load_data(sw=2700)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=0, train_size=0.8)
    y_test = keras.utils.to_categorical(y_test, num_classes=5)

    # cross validation
    kf = StratifiedKFold(n_splits=7, shuffle=False, random_state=0.3)

    for train_index, test_index in kf.split(x_train, y_train):
        train_x, train_y = x_train[train_index], y_train[train_index]
        test_x, test_y = x_train[test_index], y_train[test_index]

        train_y = keras.utils.to_categorical(train_y, num_classes=5)
        test_y = keras.utils.to_categorical(test_y, num_classes=5)

        # define neural network
        model = Sequential()
        model.add(Conv1D(32, 5, border_mode='same', input_shape=(2700, 1)))  # sliding window = 2700
        model.add(MaxPooling1D(pool_size=5, strides=None, padding='valid'))
        model.add(Dropout(0.3))
        model.add(Conv1D(64, 10, border_mode='same'))
        model.add(MaxPooling1D(pool_size=5, strides=None, padding='valid'))
        model.add(Dropout(0.4))
        model.add(Conv1D(128, 15, border_mode='same'))
        model.add(MaxPooling1D(pool_size=5, strides=None, padding='valid'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(5, activation='softmax'))
        print model.summary()

        adamoptimizer = keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='mean_squared_error', optimizer=adamoptimizer, metrics=['accuracy'])

        history = LossHistory()
        # training model
        model.fit(np.expand_dims(train_x, axis=2), train_y,
                  validation_data=(np.expand_dims(test_x, 2), test_y),
                  batch_size=512,
                  nb_epoch=500,  # training epoch
                  callbacks=[history])

        history.loss_plot('epoch')

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        n_class = [1, 2, 3, 5, 12]
        n_classes = len(n_class)

        test_y_pred = model.predict_classes(np.expand_dims(test_x, 2))
        y_pred_val = model.predict_classes(np.expand_dims(x_test, 2))
        y_pred_prob_val = model.predict_proba(np.expand_dims(x_test, 2))

        for k in range(n_classes):
            fpr[k], tpr[k], _ = roc_curve(y_test[:, k], y_pred_prob_val[:, k])
            roc_auc[k] = auc(fpr[k], tpr[k])

        lw = 2
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[k] for k in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for h in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[h], tpr[h])
        # Finally average it and comput AUC
        mean_tpr /= n_classes
        # PLot all ROC curves
        plt.figure()

        colors = cycle(['c', 'darkorange', 'cornflowerblue', 'b', 'y'])
        for p, color in zip(range(n_classes), colors):
            plt.plot(fpr[p], tpr[p], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.05f})'
                                                               ''.format(n_class[p], roc_auc[p]))

        # plt.plot(fpr[2],tpr[2],color ='darkorange',lw=lw,label='ROC curve (area = %0.2f)'%roc_auc[2])
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc='lower right')
        plt.savefig("Receiver operating characteristic_Global.tif", dpi=400)
        plt.show()

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        n_class = [1, 2, 3, 5, 12]
        n_classes = len(n_class)

        test_y_pred = model.predict_classes(np.expand_dims(test_x, 2))
        y_pred_val = model.predict_classes(np.expand_dims(x_test, 2))
        y_pred_prob_val = model.predict_proba(np.expand_dims(x_test, 2))

        for k in range(n_classes):
            fpr[k], tpr[k], _ = roc_curve(y_test[:, k], y_pred_prob_val[:, k])
            roc_auc[k] = auc(fpr[k], tpr[k])

        lw = 2
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[k] for k in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for h in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[h], tpr[h])
        # Finally average it and comput AUC
        mean_tpr /= n_classes
        # PLot all ROC curves
        plt.figure()

        colors = cycle(['c', 'darkorange', 'cornflowerblue', 'b', 'y'])
        for p, color in zip(range(n_classes), colors):
            plt.plot(fpr[p], tpr[p], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.05f})'
                                                               ''.format(n_class[p], roc_auc[p]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 0.2])
        plt.ylim([0.8, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc='lower right')
        plt.savefig("Receiver operating characteristic_local.tif", dpi=400)
        plt.show()


        def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

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


        # confusion matrix on validation set
        class_names = label_map.keys()
        cnf_matrix_val = confusion_matrix(np.argmax(test_y, axis=1), test_y_pred)
        print cnf_matrix_val
        np.set_printoptions(precision=2)
        # plot confusion matrix Number
        plt.figure()
        plot_confusion_matrix(cnf_matrix_val, classes=class_names, title='Confusion Matrix Number')
        # Plot confusion matrix probability
        plt.figure()
        plot_confusion_matrix(cnf_matrix_val, classes=class_names, normalize=True, title='Confusion Matrix Probablity')
        plt.savefig("Confusion Matrix Number_Train.tif", dpi=400)
        plt.show()

        # 测试集混淆矩阵
        cnf_matrix_test = confusion_matrix(np.argmax(y_test, axis=1), y_pred_val)
        print cnf_matrix_test
        np.set_printoptions(precision=2)
        # plot confusion matrix Number
        plt.figure()
        plot_confusion_matrix(cnf_matrix_test, classes=class_names, title='Confusion Matrix Number Test')
        # Plot confusion matrix probability
        plt.figure()
        plot_confusion_matrix(cnf_matrix_test, classes=class_names, normalize=True,
                              title='Confusion Matrix Probablity Test')
        plt.savefig("Confusion Matrix Number_Test.tif", dpi=400)
        plt.show()

        # evaluation
        acc_score = accuracy_score(np.argmax(test_y, axis=1), test_y_pred)
        print "acc_score_val {}".format(acc_score)
        acc_score_test = accuracy_score(np.argmax(y_test, axis=1), y_pred_val)
        print "acc_score_test {}".format(acc_score_test)

        auc_score_test = roc_auc_score(y_test, y_pred_prob_val)
        print "auc_score_test {}".format(auc_score_test)

        metrics_micro_val = precision_score(np.argmax(test_y, axis=1), test_y_pred, labels=n_class, average='micro')
        print "metrics_val {}".format(metrics_micro_val)
        metrics_micro_test = precision_score(np.argmax(y_test, axis=1), y_pred_val, labels=n_class, average='micro')
        print "metrics_test {}".format(metrics_micro_test)

        Cohen_kappa_score_val = cohen_kappa_score(np.argmax(test_y, axis=1), test_y_pred)
        print "Cohen_kappa_score_val {}".format(Cohen_kappa_score_val)
        Cohen_kappa_score_test = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred_val)
        print "Cohen_kappa_score_test {}".format(Cohen_kappa_score_test)

        print classification_report(np.argmax(test_y, axis=1).tolist(), test_y_pred.tolist(),
                                    target_names=[str(c) for c in class_names])
        print classification_report(np.argmax(y_test, axis=1).tolist(), y_pred_val.tolist(),
                                    target_names=[str(c) for c in class_names])

        #history.loss_plot('epoch')