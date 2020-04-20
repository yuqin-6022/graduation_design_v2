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
# 默认顺序：dloc、ed、overload-----------------------------------------------------------------------------------------
class MultiMetrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super().__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_predicts = self.model.predict(self.validation_data[0])
        task_num = len(val_predicts)

        _val_f1_all = 0

        for i in range(task_num):
            val_predict = np.argmax(val_predicts[i], -1)
            val_targ = self.validation_data[1][i]

            if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
                val_targ = np.argmax(val_targ, -1)

            _val_f1 = f1_score(val_targ, val_predict, average='weighted')
            _val_recall = recall_score(val_targ, val_predict, average='weighted')
            _val_precision = precision_score(val_targ, val_predict, average='weighted')

            _val_f1_all = _val_f1_all + _val_f1

            logs['task_%d_val_f1' % i] = _val_f1
            logs['task_%d_val_recall' % i] = _val_recall
            logs['task_%d_val_precision' % i] = _val_precision
            print(" — task_%d — val_f1: %f — val_precision: %f — val_recall: %f" % (
            i, _val_f1, _val_precision, _val_recall))

        _val_f1_mean = _val_f1_all / task_num
        logs['val_f1_mean'] = _val_f1_mean
        print(" — val_f1_mean: %f" % _val_f1_mean)

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

    model_type = 'normal'
    # model_type = 'bn_after'
    # model_type = 'mini'
    # model_type = 'nodropout'
    MODEL_PATH = os.path.join(CUR_PATH, model_type, 'best_0_mmoe.h5')

    EPOCHS = 50000
    BATCH_SIZE = 1024

    TRAIN_DIR = os.path.join(CUR_PATH, 'mmoe_train')
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)

    BEST_F1_WEIGHTS_DIR = os.path.join(TRAIN_DIR, 'mmoe_%s_best_f1' % model_type)
    if not os.path.exists(BEST_F1_WEIGHTS_DIR):
        os.makedirs(BEST_F1_WEIGHTS_DIR)

    BEST_FIT_HISTORY_DIR = os.path.join(TRAIN_DIR, 'mmoe_fit_history')
    if not os.path.exists(BEST_FIT_HISTORY_DIR):
        os.makedirs(BEST_FIT_HISTORY_DIR)

    # 数据集-----------------------------------------------------------------------------------------------------------
    train_df = pd.read_csv('../dataset/train.csv')
    test_df = pd.read_csv('../dataset/test.csv')

    TEST_SIZE = 2700

    x_train_origin = train_df.iloc[:, list(range(11))].copy().values
    y_train_origin = train_df[['dloc', 'ED', 'overload_loc']].copy().values

    x_test = test_df.iloc[:, list(range(11))].copy().values
    y_test = test_df[['dloc', 'ED', 'overload_loc']].copy().values

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_origin, y_train_origin, test_size=TEST_SIZE)

    y_dloc_train = y_train[:, 0]
    y_dloc_valid = y_valid[:, 0]
    y_dloc_test = y_test[:, 0]

    y_ED_train = y_train[:, 1]
    y_ED_valid = y_valid[:, 1]
    y_ED_test = y_test[:, 1]

    y_overload_train = y_train[:, 2]
    y_overload_valid = y_valid[:, 2]
    y_overload_test = y_test[:, 2]

    # 标准化处理-------------------------------------------------------------------------------------------------------
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    # 超参搜索开始-----------------------------------------------------------------------------------------------------
    # 考虑样本权重-----------------------------------------------------------------------------------------------------
    # my_class_weight = compute_class_weight('balanced', np.unique(y_train), y_train).tolist()
    # cw = dict(zip(np.unique(y_train), my_class_weight))
    # print(cw)

    dloc_onehot = OneHotEncoder()
    y_dloc_train = dloc_onehot.fit_transform(y_dloc_train.reshape(-1, 1)).toarray()
    y_dloc_valid = dloc_onehot.transform(y_dloc_valid.reshape(-1, 1)).toarray()
    y_dloc_test = dloc_onehot.transform(y_dloc_test.reshape(-1, 1)).toarray()

    overload_onehot = OneHotEncoder()
    y_overload_train = overload_onehot.fit_transform(y_overload_train.reshape(-1, 1)).toarray()
    y_overload_valid = overload_onehot.transform(y_overload_valid.reshape(-1, 1)).toarray()
    y_overload_test = overload_onehot.transform(y_overload_test.reshape(-1, 1)).toarray()

    # CALLBACKS = [tf.keras.callbacks.EarlyStopping(patience=3)]
    CALLBACKS = [
        MultiMetrics(valid_data=(x_valid, [y_dloc_valid, y_ED_valid, y_overload_valid]))
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.5, mode='auto')
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', patience=10, factor=0.5, mode='max')
    ]
    best_f1_weights_path = os.path.join(BEST_F1_WEIGHTS_DIR, 'f1_weight_epoch{epoch:02d}-dloc_{val_dloc_softmax_accuracy:.4f}_{task_0_val_f1:.4f}-ed_{val_ed_softmax_accuracy:.4f}_{task_1_val_f1:.4f}-overload_{val_overload_softmax_accuracy:.4f}_{task_2_val_f1:.4f}-valf1_{val_f1_mean:.4f}.hdf5')
    FIT_CALLBACKS = [
        MultiMetrics(valid_data=(x_valid, [y_dloc_valid, y_ED_valid, y_overload_valid])),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.5, mode='auto'),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', patience=10, factor=0.5, mode='max'),
        tf.keras.callbacks.ModelCheckpoint(best_f1_weights_path, monitor='val_f1_mean', verbose=2, save_best_only=True, mode='max')
    ]

    model = tf.keras.models.load_model(MODEL_PATH)

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        loss_weights=model.loss_weights,
        metrics=['accuracy']
    )

    history = model.fit(x_train, [y_dloc_train, y_ED_train, y_overload_train], batch_size=BATCH_SIZE, epochs=EPOCHS,
                             validation_data=(x_valid, [y_dloc_valid, y_ED_valid, y_overload_valid]), callbacks=FIT_CALLBACKS, verbose=2)
    evaluate_result = model.evaluate(x_test, [y_dloc_test, y_ED_test, y_overload_test])
    total_loss, dloc_loss, ed_loss, overload_loss, dloc_accuracy, ed_accuracy, overload_accuracy = evaluate_result

    print('------------------------------------------------------------------------------------------------------')
    print('evaluate_result', evaluate_result)
    print('------------------------------------------------------------------------------------------------------')

    end_time = time.time()
    time_consuming = end_time - start_time
    print('Time_consuming: %d' % int(time_consuming))

    result = dict(
        time_consuming=int(time_consuming),
        history=history.history.__str__(),
        dloc_loss=float(dloc_loss),
        dloc_accuracy=float(dloc_accuracy),
        ed_loss=float(ed_loss),
        ed_accuracy=float(ed_accuracy),
        overload_loss=float(overload_loss),
        overload_accuracy=float(overload_accuracy),
    )

    history_path = os.path.join(BEST_FIT_HISTORY_DIR, 'mmoe_%s.json' % model_type)
    with open(history_path, 'w') as f:
        json.dump(result, f)

    print('Finish!')
