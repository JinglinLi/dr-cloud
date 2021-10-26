"""
Train a deep learning model to diagnose the level of diabetic
retinopathy, and save trained model.
"""

import pandas as pd
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
#from tensorflow.keras.applications import EfficientNetB7
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential#, load_model
from keras import callbacks
import config

TARGET_SIZE = (512, 512)
INPUT_SHAPE = (512, 512, 3)

N_TRAIN = 1200

EPOCHS = 20
PATIENCE = 7

NUM_UNITS = 128
DROPOUT = 0.5
OPTIMIZER = 'adam'
METRIC_ACCURACY = 'accuracy'

# read dataframes prepared for generators
traindf = pd.read_csv(f'{config.PATH_VM}/data/output/d_traindf.csv', dtype=str)

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
model.add(Dense(NUM_UNITS, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(DROPOUT))
model.add(Dense(5, activation='softmax'))
# compile
model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=[METRIC_ACCURACY])

## for further training saved model :
# d_model_name = ''
# d_model = load_model(f'{config.PATH_VM}/dr_app/{d_model_name}')

# train the model
# stop if val_loss does not increase over PATIENCE number of epochs
callback = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    #callbacks=[callback]
)

# save trained model
model.save(f'{config.PATH_VM}/dr_app/diagnosis_resnet50v2_dense128.h5')
