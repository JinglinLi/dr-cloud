from keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
INPUT_SHAPE = (512, 512, 3)


base_model = ResNet50V2(
    weights='imagenet',
    include_top=False, # remove the top dense layers
    input_shape=INPUT_SHAPE,
    pooling='avg'
)

q_model = load_model('model/imquality_resnet50v2_dense64.h5')
#d_model = load_model('model/diagnosis_resnet50v2_dense64.h5')

print('     ')
print(base_model.summary())
print('     ')

print('     ')
print(q_model.summary())
print('     ')