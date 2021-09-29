"""print model summary"""

from keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
import config

INPUT_SHAPE = (512, 512, 3)

base_model = ResNet50V2(
    weights='imagenet',
    include_top=False, # remove the top dense layers
    input_shape=INPUT_SHAPE,
    pooling='avg'
)

q_model = load_model(f'{config.PATH_VM}/dr_app/imquality_resnet50v2_dense64.h5')

print('     ')
print(base_model.summary())
print('     ')

print('     ')
print(q_model.summary())
print('     ')
