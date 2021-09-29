"""
Train a deep learning model to determine whether overall quality
of an image is sufficient for diagnosis, and save trained model.
"""

import pandas as pd
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential#, load_model
from keras import callbacks
import config


TARGET_SIZE = (512, 512)
INPUT_SHAPE = (512, 512, 3)

EPOCHS = 10
PATIENCE = 5

NUM_UNITS = 64
DROPOUT = 0.5
OPTIMIZER = 'adam'
METRIC_ACCURACY = 'accuracy'


# read dataframes prepared for generators
traindf = pd.read_csv(f'{config.PATH_VM}/data/output/q_traindf.csv', dtype='str')

# prepare data generators for fiting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=traindf,
    x_col="im_path",
    y_col="im_quality",
    subset="training",
    class_mode="binary",
    target_size=TARGET_SIZE
)

valid_generator = train_datagen.flow_from_dataframe(
    dataframe=traindf,
    x_col="im_path",
    y_col="im_quality",
    subset="validation",
    class_mode="binary",
    target_size=TARGET_SIZE
)

# model : transfer learning from pretrained ResNet50V2
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
q_model = Sequential()
q_model.add(base_model)
# add custom layer on top of base model
q_model.add(Dense(NUM_UNITS, activation='relu'))
q_model.add(BatchNormalization())
q_model.add(Dropout(DROPOUT))
q_model.add(Dense(1, activation='sigmoid'))
# compile
q_model.compile(loss='binary_crossentropy',
              optimizer=OPTIMIZER,
              metrics=[METRIC_ACCURACY])

# # for further training saved model :
# q_model_name = 'imquality_resnet50v2_dense64.h5'
# q_model = load_model(f'{config.PATH_VM}/model/{q_model_name}')

# train the model>
# stop if val_loss does not increase over PATIENCE number of epochs
callback = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
q_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=[callback])

# save trained model
q_model.save(f'{config.PATH_VM}/model/imquality_resnet50v2_dense64_v2.h5')
