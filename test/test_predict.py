""" test dr_app/predict.py """

import pytest
from PIL import Image
from dr_app.predict import Predict
import config

# load an example image
im = Image.open(f'{config.PATH_DISK}/data/kaggle/test_images/ffdc2152d455.png')
pred_im = Predict(im)

def test_preprocess_image():
    prepro_im = pred_im.preprocess_image()
    assert prepro_im.shape == (1, 512, 512, 3)

def test_predict_dr_level():
    quality = pred_im.predict_quality()
    assert quality in Predict.C_QUALITY.values()

def predict_quality():
    diagnosis = pred_im.predict_dr_level()
    assert diagnosis in Predict.C_DIAGNOSIS.values()
