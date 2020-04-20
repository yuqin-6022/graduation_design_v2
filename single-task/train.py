#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time      : 2020/4/16 17:54
# @Author    : Shawn Li
# @FileName  : train.py
# @IDE       : PyCharm
# @Blog      : 暂无

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state, compute_class_weight
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, recall_score, precision_score
from datetime import datetime
import time
import os
import json


# MultiMetrics---------------------------------------------------------------------------------------------------------
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='weighted')
        _val_recall = recall_score(val_targ, val_predict, average='weighted')
        _val_precision = precision_score(val_targ, val_predict, average='weighted')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return


# 终端运行-------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Starting...')
    start_time = time.time()
    CUR_PATH = os.getcwd()
    DATETIME = datetime.now().strftime('%Y%m%d%H%M%S')

    # 设置gpu---------------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus)

    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
    )

    # for gpu in gpus:
    #     tf.config.experimental.set_virtual_device_configuration(
    #         gpu,
    #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
    #     )

    model_type = 'mini'
    # model_type = 'mini-bn_after'
    # model_type = 'mini-nodropout'

    y_type = 'dloc'
    # y_type = 'ED'
    # y_type = 'overload_loc'

    MODEL_PATH = os.path.join(CUR_PATH, model_type, '%s_best_dnn.h5' % y_type)

    EPOCHS = 50000
    BATCH_SIZE = 1024

    TRAIN_DIR = os.path.join(CUR_PATH, 'single_train')
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)

    BEST_F1_WEIGHTS_DIR = os.path.join(TRAIN_DIR, 'single_%s_best_f1' % model_type)
    if not os.path.exists(BEST_F1_WEIGHTS_DIR):
        os.makedirs(BEST_F1_WEIGHTS_DIR)
    BEST_F1_WEIGHTS_DIR = os.path.join(BEST_F1_WEIGHTS_DIR, 'single_%s_best_f1' % y_type)
    if not os.path.exists(BEST_F1_WEIGHTS_DIR):
        os.makedirs(BEST_F1_WEIGHTS_DIR)


    BEST_FIT_HISTORY_DIR = os.path.join(TRAIN_DIR, 'single_fit_history')
    if not os.path.exists(BEST_FIT_HISTORY_DIR):
        os.makedirs(BEST_FIT_HISTORY_DIR)
    BEST_FIT_HISTORY_DIR = os.path.join(BEST_FIT_HISTORY_DIR, 'single_%s_fit_history' % model_type)
    if not os.path.exists(BEST_FIT_HISTORY_DIR):
        os.makedirs(BEST_FIT_HISTORY_DIR)


    # 数据集-----------------------------------------------------------------------------------------------------------
    train_df = pd.read_csv('../dataset/train.csv')
    test_df = pd.read_csv('../dataset/test.csv')

    TEST_SIZE = 2700

    x_train_origin = train_df.iloc[:, list(range(11))].copy().values
    y_train_origin = train_df[y_type].copy().values

    x_test = test_df.iloc[:, list(range(11))].copy().values
    y_test = test_df[y_type].copy().values

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_origin, y_train_origin, test_size=TEST_SIZE)


    # 标准化处理-------------------------------------------------------------------------------------------------------
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    # 超参搜索开始-----------------------------------------------------------------------------------------------------
    # 考虑样本权重-----------------------------------------------------------------------------------------------------
    my_class_weight = compute_class_weight('balanced', np.unique(y_train), y_train).tolist()
    cw = dict(zip(np.unique(y_train), my_class_weight))
    print(cw)

    if y_type != 'ED':
        onehot = OneHotEncoder()
        y_train = onehot.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_valid = onehot.transform(y_valid.reshape(-1, 1)).toarray()
        y_test = onehot.transform(y_test.reshape(-1, 1)).toarray()

    # CALLBACKS = [tf.keras.callbacks.EarlyStopping(patience=3)]
    best_f1_weights_path = os.path.join(BEST_F1_WEIGHTS_DIR, '%s_f1_weight_epoch{epoch:02d}-valacc{val_accuracy:.4f}-valf1{val_f1:.4f}.hdf5' % y_type)
    CALLBACKS = [
        Metrics(valid_data=(x_valid, y_valid)),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.5, mode='auto'),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', patience=10, factor=0.5, mode='max'),
        tf.keras.callbacks.ModelCheckpoint(best_f1_weights_path, monitor='val_f1', verbose=2, save_best_only=True, mode='max')
    ]

    model = tf.keras.models.load_model(MODEL_PATH)

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=['accuracy']
    )

    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, class_weight=cw,
                             validation_data=(x_valid, y_valid), callbacks=CALLBACKS, verbose=2)
    evaluate_result = model.evaluate(x_test, y_test)
    test_loss, test_accuracy = evaluate_result

    print('------------------------------------------------------------------------------------------------------')
    print('evaluate_result', evaluate_result)
    print('------------------------------------------------------------------------------------------------------')

    end_time = time.time()
    time_consuming = end_time - start_time
    print('Time_consuming: %d' % int(time_consuming))

    result = dict(
        time_consuming=int(time_consuming),
        history=history.history.__str__(),
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy),
    )

    history_path = os.path.join(BEST_FIT_HISTORY_DIR, '%s.json' % y_type)
    with open(history_path, 'w') as f:
        json.dump(result, f)

    print('Finish!')
