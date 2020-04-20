#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time      : 2020/4/15 18:13
# @Author    : Shawn Li
# @FileName  : single_task_hp.py
# @IDE       : PyCharm
# @Blog      : 暂无

import tensorflow as tf
import numpy as np
import pandas as pd
import kerastuner
from kerastuner import HyperModel
from kerastuner.tuners.bayesian import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state, compute_class_weight
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, recall_score, precision_score
from datetime import datetime
import time
import os
import json


# Metrics--------------------------------------------------------------------------------------------------------------
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


# MyHyperModel---------------------------------------------------------------------------------------------------------
class MyHyperModel(HyperModel):
    def __init__(self, input_shape, output_num, loss):
        super().__init__()
        self.input_shape = input_shape
        self.output_num = output_num
        self.loss = loss

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        for i in range(hp.Int('num_layers', min_value=1, max_value=5, step=1)):
            model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                         min_value=16,
                                                         max_value=256,
                                                         step=16)))
            model.add(tf.keras.layers.BatchNormalization())  # 先bn
            model.add(tf.keras.layers.Activation('relu'))  # 再激活函数
            model.add(tf.keras.layers.Dropout(rate=hp.Float('rate_' + str(i),
                                                            min_value=0,
                                                            max_value=0.75,
                                                            step=0.05)))
        # 输出层
        # model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(hp.Choice('l2_rate', [1e-2, 1e-3, 1e-4]))))
        model.add(tf.keras.layers.Dense(units=self.output_num, activation='softmax'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss=self.loss,
            metrics=['accuracy']
        )

        return model


# 终端运行-------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Starting...')
    start_time = time.time()

    y_type = 'dloc'
    # y_type = 'ED'
    # y_type = 'overload_loc'

    # 设置gpu---------------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus)

    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
    )


    # for gpu in gpus:
    #     tf.config.experimental.set_virtual_device_configuration(
    #         gpu,
    #         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
    #     )

    OBJECTIVE = kerastuner.Objective("val_f1", direction="max")
    # OBJECTIVE = 'val_accuracy'
    MAX_TRIALS = 25
    EPOCHS = 10000
    BATCH_SIZE = 1024
    CUR_PATH = os.getcwd()
    DATETIME = datetime.now().strftime('%Y%m%d%H%M%S')

    EXPERIMENT_PREFIX = y_type

    KERAS_TUNER_DIR = os.path.join(CUR_PATH, 'keras_tuner_dir')
    if not os.path.exists(KERAS_TUNER_DIR):
        os.makedirs(KERAS_TUNER_DIR)
    KERAS_TUNER_DIR = os.path.join(KERAS_TUNER_DIR, '%s_keras_tuner_dir_%s' % (EXPERIMENT_PREFIX, DATETIME))
    os.makedirs(KERAS_TUNER_DIR)

    KERAS_TUNER_MODEL_DIR = os.path.join(CUR_PATH, 'keras_tuner_models')
    if not os.path.exists(KERAS_TUNER_MODEL_DIR):
        os.makedirs(KERAS_TUNER_MODEL_DIR)
    KERAS_TUNER_MODEL_DIR = os.path.join(KERAS_TUNER_MODEL_DIR, '%s_keras_tuner_models_%s' % (EXPERIMENT_PREFIX, DATETIME))
    os.makedirs(KERAS_TUNER_MODEL_DIR)

    BEST_F1_WEIGHTS_DIR = os.path.join(CUR_PATH, '%s_best_f1_%s' % (EXPERIMENT_PREFIX, DATETIME))
    if not os.path.exists(BEST_F1_WEIGHTS_DIR):
        os.makedirs(BEST_F1_WEIGHTS_DIR)

    BEST_FIT_HISTORY_DIR = os.path.join(CUR_PATH, '%s_best_fit_histories_%s' % (EXPERIMENT_PREFIX, DATETIME))
    if not os.path.exists(BEST_FIT_HISTORY_DIR):
        os.makedirs(BEST_FIT_HISTORY_DIR)

    # 数据集-----------------------------------------------------------------------------------------------------------
    train_df = pd.read_csv('../../dataset/train.csv')
    test_df = pd.read_csv('../../dataset/test.csv')

    print('--------------------------------------------------------------------------------------------------------')
    print(y_type)
    print('--------------------------------------------------------------------------------------------------------')

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

    my_loss = 'sparse_categorical_crossentropy'
    if y_type != 'ED':
        onehot = OneHotEncoder()
        y_train = onehot.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_valid = onehot.transform(y_valid.reshape(-1, 1)).toarray()
        y_test = onehot.transform(y_test.reshape(-1, 1)).toarray()
        my_loss = 'categorical_crossentropy'

    # keras-tuner部分设置----------------------------------------------------------------------------------------------
    # CALLBACKS = [tf.keras.callbacks.EarlyStopping(patience=3)]
    CALLBACKS = [
        Metrics(valid_data=(x_valid, y_valid))
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.5, mode='auto')
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', patience=10, factor=0.5, mode='max')
    ]
    best_f1_weights_path = os.path.join(BEST_F1_WEIGHTS_DIR, '%s_f1_weight_epoch{epoch:02d}-valacc{val_accuracy:.4f}-valf1{val_f1:.4f}.hdf5' % y_type)
    FIT_CALLBACKS = [
        Metrics(valid_data=(x_valid, y_valid)),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=10, factor=0.5, mode='auto'),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', patience=10, factor=0.5, mode='max'),
        tf.keras.callbacks.ModelCheckpoint(best_f1_weights_path, monitor='val_f1', verbose=2, save_best_only=True, mode='max')
    ]
    PROJECT_NAME = '%s_single_dnn_keras_tuner_%s' % (y_type, DATETIME)

    # 实例化贝叶斯优化器
    y_num = len(train_df[y_type].unique()) + (y_type == 'ED')  # ED从1~9开始，所以补一个输出
    hypermodel = MyHyperModel((x_train.shape[1],), y_num, my_loss)
    tuner = BayesianOptimization(hypermodel, objective=OBJECTIVE, max_trials=MAX_TRIALS, directory=KERAS_TUNER_DIR, project_name=PROJECT_NAME)
    # 开始计时超参数搜索
    tuner_start_time = datetime.now()
    tuner_start = time.time()
    # 开始超参数搜索
    tuner.search(x_train, y_train, class_weight=cw, batch_size=BATCH_SIZE, epochs=EPOCHS,
                 validation_data=(x_valid, y_valid), callbacks=CALLBACKS)
    # tuner.search(x_train, y_train, batch_size=TUNER_BATCH_SIZE, epochs=TUNER_EPOCHS, validation_data=(x_valid, y_valid))
    # 结束计时超参数搜索
    tuner_end_time = datetime.now()
    tuner_end = time.time()
    # 统计超参数搜索用时
    tuner_duration = tuner_end - tuner_start

    # 获取前BEST_NUM个最优超参数--------------------------------------------------------------
    best_models = tuner.get_best_models()
    best_model = best_models[0]
    best_model_path = os.path.join(KERAS_TUNER_MODEL_DIR, '%s_best_dnn.h5' % y_type)
    best_model.save(best_model_path)

    history = best_model.fit(x_train, y_train, class_weight=cw, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_valid, y_valid), callbacks=FIT_CALLBACKS, verbose=2)
    evaluate_result = best_model.evaluate(x_test, y_test)
    test_loss, test_accuracy = evaluate_result

    print('------------------------------------------------------------------------------------------------------')
    print('%s_test_result: ' % y_type)
    print('test_loss: %.4f, test_accuracy: %.4f' % (test_loss, test_accuracy))
    print('------------------------------------------------------------------------------------------------------')

    end_time = time.time()
    time_consuming = end_time - start_time
    print('Time_consuming: %d' % int(time_consuming))

    result = dict(
        time_consuming=int(time_consuming),
        history=history.history.__str__(),
        test_loss=float(test_loss),
        test_accuracy=float(test_accuracy)
    )

    history_path = os.path.join(BEST_FIT_HISTORY_DIR, '%s.json' % (y_type))
    with open(history_path, 'w') as f:
        json.dump(result, f)

    print('Finish!')

