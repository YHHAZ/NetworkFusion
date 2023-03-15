from __future__ import absolute_import, print_function

import json
import os
import sys

import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras_applications.inception_resnet_v2 import InceptionResNetV2
from keras_applications.inception_v3 import InceptionV3
from keras_applications.nasnet import NASNetLarge
from keras_applications.resnext import ResNeXt101
from keras_applications.xception import Xception
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (Activation, Add, AveragePooling2D,
                                     BatchNormalization, Concatenate, Conv2D,
                                     Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Input, Lambda,
                                     LeakyReLU, Permute, Reshape, multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adagrad, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

import sys
name_work1 = sys.argv[1]
work_port1 = sys.argv[2]
name_work2 = sys.argv[3]
work_port2 = sys.argv[4]
name_work3 = sys.argv[5]
work_port3 = sys.argv[6]
name_work4 = sys.argv[7]
work_port4 = sys.argv[8]
name_work5 = sys.argv[9]
work_port5 = sys.argv[10]
index_work = sys.argv[11]
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': [
            str(name_work1) + str(":") + str(work_port1),
            str(name_work2) + str(":") + str(work_port2),
            str(name_work3) + str(":") + str(work_port3),
            str(name_work4) + str(":") + str(work_port4),
            str(name_work5) + str(":") + str(work_port5)
        ]
    },
    'task': {
        'type': 'worker',
        'index': int(index_work)
    }
})
print(os.environ['TF_CONFIG'])
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
)

with strategy.scope():

    def cbam_block(cbam_feature, LayerName='', ratio=8):
        """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        As described in https://arxiv.org/abs/1807.06521.
        """

        cbam_feature = channel_attention(cbam_feature, ratio, LayerName)
        cbam_feature = spatial_attention(cbam_feature, LayerName)
        return cbam_feature

    def channel_attention(input_feature, ratio=8, LayerName=""):

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]

        shared_layer_one = Dense(channel // ratio,
                                 activation='relu',
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros',
                                 name=LayerName + 'Dense_1')
        shared_layer_two = Dense(channel,
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros',
                                 name=LayerName + 'Dense_2')

        avg_pool = GlobalAveragePooling2D(
            name=LayerName + 'GlobalAveragePooling2D_1')(input_feature)
        avg_pool = Reshape((1, 1, channel),
                           name=LayerName + 'Reshape_1')(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel // ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel)

        max_pool = GlobalMaxPooling2D(
            name=LayerName + 'GlobalAveragePooling2D_2')(input_feature)
        max_pool = Reshape((1, 1, channel),
                           name=LayerName + 'Reshape_2')(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel // ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel)

        cbam_feature = Add(name=LayerName + 'Add')([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid',
                                  name=LayerName + 'Activation')(cbam_feature)

        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)
        print(input_feature.shape, cbam_feature.shape,
              multiply([input_feature, cbam_feature]).shape)
        return multiply([input_feature, cbam_feature])

    def spatial_attention(input_feature, LayerName):
        kernel_size = 7

        if K.image_data_format() == "channels_first":
            channel = input_feature.shape[1]
            cbam_feature = Permute((2, 3, 1))(input_feature)
        else:
            channel = input_feature.shape[-1]
            cbam_feature = input_feature

        avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True),
                          name=LayerName + 'Lambda_1')(cbam_feature)
        assert avg_pool.shape[-1] == 1
        max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True),
                          name=LayerName + 'Lambda_2')(cbam_feature)
        assert max_pool.shape[-1] == 1
        concat = Concatenate(axis=3, name=LayerName +
                             'Concatenate')([avg_pool, max_pool])
        assert concat.shape[-1] == 2
        cbam_feature = Conv2D(filters=1,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False,
                              name=LayerName + 'Conv2D_1')(concat)
        assert cbam_feature.shape[-1] == 1

        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)
        print(input_feature.shape, cbam_feature.shape,
              multiply([input_feature, cbam_feature]).shape)
        return multiply([input_feature, cbam_feature])

    def InceptionV3_build_model(inputs_dim,
                                out_dims,
                                num_classes,
                                input_shape=(512, 512, 3)):
        x = InceptionV3(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=input_shape,
                        pooling=max,
                        backend=keras.backend,
                        layers=keras.layers,
                        models=keras.models,
                        utils=keras.utils)(inputs_dim)

        scale = cbam_block(x, 'build_model_inceptionV3_')

        x = GlobalAveragePooling2D(
            name='build_model_inceptionV3_main_GlobalAveragePooling2D')(scale)
        dp_1 = Dropout(0.6, name='build_model_inceptionV3_main_Dropout')(x)
        fc2 = Dense(out_dims,
                    kernel_initializer='he_normal',
                    name='build_model_inceptionV3_main_Dense_1')(dp_1)
        fc2 = LeakyReLU(alpha=0.0001,
                        name='build_model_inceptionV3_main_LeakyReLU')(fc2)
        fc2_num_classes = Dense(
            num_classes,
            kernel_initializer='he_normal',
            name='build_model_inceptionV3_main_Dense_2')(dp_1)
        fc2_num_classes = Activation(
            'softmax', name='build_model_inceptionV3')(fc2_num_classes)
        return fc2, fc2_num_classes

    def InceptionResNetV2_build_model(inputs_dim,
                                      out_dims,
                                      num_classes,
                                      input_shape=(512, 512, 3)):
        x = InceptionResNetV2(include_top=False,
                              weights='imagenet',
                              input_tensor=None,
                              input_shape=input_shape,
                              pooling=max,
                              backend=keras.backend,
                              layers=keras.layers,
                              models=keras.models,
                              utils=keras.utils)(inputs_dim)

        scale = cbam_block(x, 'build_model_inceptionResNetV2_')
        x = GlobalAveragePooling2D(
            name='build_model_inceptionResNetV2_main_GlobalAveragePooling2D')(
                scale)
        dp_1 = Dropout(0.6,
                       name='build_model_inceptionResNetV2_main_Dropout')(x)
        fc2 = Dense(out_dims,
                    kernel_initializer='he_normal',
                    name='build_model_inceptionResNetV2_main_Dense_1')(dp_1)
        fc2 = LeakyReLU(
            alpha=0.0001,
            name='build_model_inceptionResNetV2_main_LeakyReLU')(fc2)

        fc2_num_classes = Dense(
            num_classes,
            kernel_initializer='he_normal',
            name='build_model_inceptionResNetV2_main_Dense_2')(dp_1)
        fc2_num_classes = Activation(
            'softmax', name='build_model_inceptionResNetV2')(fc2_num_classes)
        return fc2, fc2_num_classes

    def Xception_build_model(inputs_dim,
                             out_dims,
                             num_classes,
                             input_shape=(512, 512, 3)):
        x = Xception(include_top=False,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=input_shape,
                     pooling=max,
                     backend=keras.backend,
                     layers=keras.layers,
                     models=keras.models,
                     utils=keras.utils)(inputs_dim)

        scale = cbam_block(x, 'build_model_xception')
        x = GlobalAveragePooling2D(
            name='build_model_xception_main_GlobalAveragePooling2D')(scale)
        dp_1 = Dropout(0.6, name='build_model_xception_main_Dropout')(x)
        fc2 = Dense(out_dims,
                    kernel_initializer='he_normal',
                    name='build_model_xception_main_Dense_1')(dp_1)
        fc2 = LeakyReLU(alpha=0.0001,
                        name='build_model_xception_main_LeakyReLU')(fc2)

        fc2_num_classes = Dense(num_classes,
                                kernel_initializer='he_normal',
                                name='build_model_xception_main_Dense_2')(dp_1)
        fc2_num_classes = Activation(
            'softmax', name='build_model_xception')(fc2_num_classes)
        return fc2, fc2_num_classes

    def build_model(num_classes, input_shape):
        out_dims = 10
        inputs_dim = Input(input_shape)
        InceptionV3_build_model1, InceptionV3_num_classes = InceptionV3_build_model(
            inputs_dim, out_dims, num_classes, (height, width, channels))
        InceptionResNetV2_build_model2, InceptionResNetV2_num_classes = InceptionResNetV2_build_model(
            inputs_dim, out_dims, num_classes, (height, width, channels))
        Xception_build_model3, Xception_num_classes = Xception_build_model(
            inputs_dim, out_dims, num_classes, (height, width, channels))

        model = Concatenate(name='build_model_main_Concatenate')([
            InceptionV3_build_model1, InceptionResNetV2_build_model2,
            Xception_build_model3
        ])

        model = Dense(num_classes,
                      kernel_initializer='he_normal',
                      name='build_model_main_Dense')(model)
        All_num_classes = Activation('softmax',
                                     name='build_model_All_num_classes')(
                                         model)  #此处注意，为sigmoid函数
        model = Model(inputs=inputs_dim,
                      outputs=[
                          InceptionV3_num_classes,
                          InceptionResNetV2_num_classes, Xception_num_classes,
                          All_num_classes
                      ])
        return model

    def setup_to_fine_tune_1(model):
        LayersNum = 0
        for layer in model.layers:
            if not layer.name.startswith('build_model'):
                layer.trainable = False
                LayersNum += 1
        print('不可以训练的层有: ' + str(LayersNum) + "可以训练的层有: " +
              str(len(model.layers) - LayersNum))
        loss = {
            'build_model_inceptionV3': 'categorical_crossentropy',
            'build_model_inceptionResNetV2': 'categorical_crossentropy',
            'build_model_xception': 'categorical_crossentropy',
            'build_model_All_num_classes': 'categorical_crossentropy'
        }
        loss_weights = {
            'build_model_inceptionV3': 1,
            'build_model_inceptionResNetV2': 1,
            'build_model_xception': 1,
            'build_model_All_num_classes': 1
        }

        model.compile(optimizer=Adam(lr=0.01),
                      loss=loss,
                      loss_weights=loss_weights,
                      metrics=['accuracy'])

    def setup_to_fine_tune_2(model):
        LayersNum = 0
        for layer in model.layers:
            layer.trainable = True
            LayersNum += 1
        print('不可以训练的层有: ' + str(LayersNum) + "可以训练的层有: " +
              str(len(model.layers) - LayersNum))
        loss = {
            'InceptionV3_build_model': 'categorical_crossentropy',
            'InceptionResNetV2_build_model': 'categorical_crossentropy',
            'Xception_build_model': 'categorical_crossentropy',
            'All_num_classes': 'categorical_crossentropy'
        }
        loss_weights = {
            'InceptionV3_build_model': 1,
            'InceptionResNetV2_build_model': 1,
            'Xception_build_model': 1,
            'All_num_classes': 1
        }
        model.compile(optimizer=Adam(lr=0.00001),
                      loss=loss,
                      loss_weights=loss_weights,
                      metrics=['accuracy'])

    height = 450  #图片的高度
    width = 600  #图片的长度
    channels = 3  #彩色图片
    batch_size = 150 * 5 
    num_classes = 7 
    SEED = 666
    epochs = 300

    train_dir = "/data/azhuang/Skin/HAM10000/Image_transfer/trainImage"
    valid_dir = "/data/azhuang/Skin/HAM10000/Image_transfer/validImage"

    train_datagen = keras.preprocessing.image.ImageDataGenerator(  
        rescale=1. / 255, ) 

    train_generator = train_datagen.flow_from_directory(
        train_dir,  #上面的ImageDataGenerator只是一个迭代器，将图片转化成像素值，这个方法flow_from_directory就可以批量取数据
        target_size=(height, width),  #图片大小规定到这个高宽
        batch_size=batch_size,  #每一个批次batch_size个图片进行上面的操作
        seed=SEED,
        shuffle=True,
        class_mode="categorical")  #这个指定二进制标签，我们用了binary_crossentropy损失函数
    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)  #验证集不用添加图片，只需要将图片像素值进行规定
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(height, width),
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode="categorical")
    train_num = train_generator.samples  #获取训练样本总数
    valid_num = valid_generator.samples
    print("样本总数为：")
    print(train_num, valid_num)

    model = build_model(num_classes, (height, width, channels))
    setup_to_fine_tune_1(model)

    import os
    logdir = './callbacks_EarlyStopping_1'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir, "callbacks_EarlyStopping.h5")
    log_dir = os.path.join('log_1')  #win10下的bug，
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    #回调函数的使用-在训练中数据的保存
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            output_model_file,  #最后模型的保存-加上下面的代码代表就是最优模型的保存
            monitor='val_loss',
            save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='auto', min_delta=1e-10, patience=11
        ),  #如果模型提前关闭的参数设置，patience参数的意义在于:当迭代次数5次检测指标的值都是比我规定的是小的话，就直接停止模型的训练
        #min_delta参数的意思就是:本次训练的测试指标的值与上一次的值的差值是不是比这个阈值要低，如果低的话就停止模型的训练
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                          patience=5,
                                          mode='auto',
                                          verbose=1,
                                          min_delta=1e-9),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    history = model.fit_generator(
        train_generator,  #迭代器对象
        steps_per_epoch=train_num // batch_size,  #因为迭代器是无限次的，所以要规定什么时候退出
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_num // batch_size,
        callbacks=callbacks)
    print('Saving model to disk\n')
    model.save('model_Deep_ensemble_learning_1_{}.h5'.format(index_work))
    print("history保存")
    import pickle
    with open('model_Deep_ensemble_learning_1_{}.pickle'.format(index_work),
              'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    height = 450  #图片的高度
    width = 600  #图片的长度
    channels = 3  #彩色图片
    batch_size = 150 * 5  
    num_classes = 7
    SEED = 666
    epochs = 300

    train_dir = "/data/azhuang/Skin/HAM10000/Image_transfer/trainImage"
    valid_dir = "/data/azhuang/Skin/HAM10000/Image_transfer/validImage"

    train_datagen = keras.preprocessing.image.ImageDataGenerator(  
        rescale=1. / 255, )  

    train_generator = train_datagen.flow_from_directory(
        train_dir,  #上面的ImageDataGenerator只是一个迭代器，将图片转化成像素值，这个方法flow_from_directory就可以批量取数据
        target_size=(height, width),  #图片大小规定到这个高宽
        batch_size=batch_size,  #每一个批次batch_size个图片进行上面的操作
        seed=SEED,
        shuffle=True,
        class_mode="categorical")  #这个指定二进制标签，我们用了binary_crossentropy损失函数
    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)  #验证集不用添加图片，只需要将图片像素值进行规定
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(height, width),
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode="categorical")
    train_num = train_generator.samples  #获取训练样本总数
    valid_num = valid_generator.samples
    print("样本总数为：")
    print(train_num, valid_num)

    model = load_model(
        '/data/azhuang/Skin/HAM10000/Image_transfer/callbacks_EarlyStopping_Deep_ensemble_learning/callbacks_EarlyStopping.h5'
    )
    print(model.summary())
    with open('modelSummary.txt', 'w') as f:
        f.write(str(model.summary()))
    setup_to_fine_tune_2(model)

    import os
    logdir = './callbacks_EarlyStopping_2'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir, "callbacks_EarlyStopping.h5")
    log_dir = os.path.join('log_2')  #win10下的bug
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    #回调函数的使用-在训练中数据的保存
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            output_model_file,  #最后模型的保存-加上下面的代码代表就是最优模型的保存
            monitor='val_All_num_classes_accuracy',
            save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor='val_All_num_classes_accuracy',
            mode='auto',
            min_delta=1e-10,
            patience=11
        ),  #如果模型提前关闭的参数设置，patience参数的意义在于:当迭代次数5次检测指标的值都是比我规定的是小的话，就直接停止模型的训练
        #min_delta参数的意思就是:本次训练的测试指标的值与上一次的值的差值是不是比这个阈值要低，如果低的话就停止模型的训练
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_All_num_classes_accuracy',
            patience=5,
            mode='auto',
            verbose=1,
            min_delta=1e-9),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    history = model.fit_generator(
        train_generator,  #迭代器对象
        steps_per_epoch=train_num // batch_size,  #因为迭代器是无限次的，所以要规定什么时候退出
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=valid_num // batch_size,
        callbacks=callbacks)
    print('Saving model to disk\n')
    model.save('model_Deep_ensemble_learning_2_{}.h5'.format(index_work))
    print("history保存")
    import pickle
    with open('model_Deep_ensemble_learning_2_{}.pickle'.format(index_work),
              'wb') as file_pi:
        pickle.dump(history.history, file_pi)
