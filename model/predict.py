"""
load trained model to predict image quality and diabetic retinopathy level
of given retina image

image quality :
0 : bad
1 : good

diabetic retinopathy level
0 : 'No DR'
1 : 'Mild'
2 : 'Moderate'
3 : 'Severe'
4 : 'Proliferative DR'
"""

import numpy as np
from PIL import Image
from skimage import transform
from keras.models import load_model
#import config


class Predict:
    """
    predict diabetic retinopathy level based on input image
    """

    # quality model and quality classes
    #M_QUALITY = load_model(f'{config.PATH_VM}/model/imquality_resnet50v2_dense64.h5')
    M_QUALITY = load_model('./model/imquality_resnet50v2_dense64.h5')
    C_QUALITY = {
        0 : 'Quality is `not good` enough for the diagnosis of retinal diseases',
        1 : 'Quality is `good` enough for the diagnosis of retinal diseases'}

    # diagnosis model and quality classes
    #M_DIAGNOSIS = load_model(f'{config.PATH_VM}/model/diagnosis_resnet50v2_dense128.h5')
    M_QUALITY = load_model('./model/imquality_resnet50v2_dense64.h5')
    C_DIAGNOSIS = {
        0 : 'No DR : No apparent retinopathy, no visible sign of abnormalities',
        1 : 'Mild – NPDR : Only presence of Microaneurysms',
        2 : 'Moderate – NPDR : More than just microaneurysms but less than severe NPDR',
        3 : """Severe – NPDR : Moderate NPDR and any of the following:
            - over 20 intraretinal hemorrhages
            - Venous beading
            - Intraretinal microvascular abnormalities
            - No signs of PDR""",
        4 : """PDR Severe NPDR and one or both of the following:
            - Neovascularization
            - Vitreous/preretinal hemorrhage"""}

    def __init__(self, image):
        self.image = image
        self.pred_quality = '' # descriptive string
        self.pred_quality_details = [] # list of probabilities
        self.pred_diagnosis = '' # descriptive string
        self.pred_diagnosis_details = [] # list of probabilities

    def preprocess_image(self):
        """image preprocessing for MobileNetV2"""
        pp_image = np.array(self.image).astype('float32')/255
        pp_image = transform.resize(pp_image, (512, 512, 3))
        pp_image = np.expand_dims(pp_image, axis=0)
        return pp_image

    def predict_quality(self):
        """predict input image quality"""
        pp_image = self.preprocess_image()
        self.pred_quality_details = Predict.M_QUALITY.predict(pp_image)[0][0]
        self.pred_quality = Predict.C_QUALITY.get(np.round(self.pred_quality_details))
        print(self.pred_quality, self.pred_quality_details)
        # print(str(self.pred_quality_details))
        return self.pred_quality

    def predict_dr_level(self):
        """predict diabetic retinopathy level based on input image"""
        pp_image = self.preprocess_image()
        self.pred_diagnosis_details = Predict.M_DIAGNOSIS.predict(pp_image)
        max_ind = np.argmax(self.pred_diagnosis_details)
        print(max_ind)
        self.pred_diagnosis = Predict.C_DIAGNOSIS.get(max_ind)
        print(self.pred_diagnosis, self.pred_diagnosis_details)
        return self.pred_diagnosis

if __name__ == '__main__':
    im = Image.open(f'{config.PATH_DISK}/data/kaggle/test_images/ffdc2152d455.png')
    p = Predict(im)
    p.predict_quality()
    p.predict_dr_level()
