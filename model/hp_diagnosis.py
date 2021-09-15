import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import pandas as pd
import numpy as np
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
#from tensorflow.keras.applications import EfficientNetB7
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential # load model
from keras import callbacks
from sklearn.metrics import accuracy_score
import config


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32, 64, 128]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.4))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def train_test_model(hparams):
    """train and evaluate model"""
    TARGET_SIZE = (512, 512)
    INPUT_SHAPE = (512, 512, 3)
    EPOCHS = 1
    PATIENCE = 1
    #PATH = '/Users/jinglin/Documents/spiced_projects/dr_app'
    PATH = '/mnt'

    ############ train, validation, test generators 
    # read dataframes prepared for generators
    traindf = pd.read_csv(f'{config.PATH_VM}/data/output/d_traindf.csv', dtype=str)
    testdf = pd.read_csv(f'{config.PATH_VM}/data/output/d_testdf.csv', dtype=str)
    # prepare train, valid, test generator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=traindf,
        x_col="im_path",
        y_col="diagnosis",
        subset='training',
        class_mode='categorical',
        target_size=TARGET_SIZE
    )
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=traindf,
        x_col="im_path",
        y_col="diagnosis",
        subset='validation',
        class_mode='categorical',
        target_size=TARGET_SIZE
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=testdf,
        x_col="im_path",
        y_col=None,
        class_mode=None,
        target_size=TARGET_SIZE
    )

    ################# model
    # iniciate and compile model : transfer learning from pretrained ResNet50V2
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False, # remove the top dense layers
        input_shape=INPUT_SHAPE,
        pooling='avg'
    )
    # freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False
    # generate sequential model with pretrained base model
    model = Sequential()
    model.add(base_model)
    # add custom layer on top of base model
    model.add(Dense(hparams[HP_NUM_UNITS], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(5, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy',
                optimizer=hparams[HP_OPTIMIZER],
                metrics=[METRIC_ACCURACY])

    ############## train
    # stop if val_loss does not increase over PATIENCE number of epochs
    callback = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[callback]
    )

    ############# evaluate with test
    y_true = testdf['diagnosis']
    y_pred_detail = model.predict_generator(test_generator)
    y_pred = [str(np.argmax(y_pred_i)) for y_pred_i in y_pred_detail]
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1