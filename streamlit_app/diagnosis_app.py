""" DIAGNOSIS APP """ 
import streamlit as st
from PIL import Image
import numpy as np
from model.predict import Predict

def app():
    """diagnosis app"""

    st.header('DIABETIC RETINOPATHY DIAGNOSTIC APP')

    col1, col2 = st.columns(2)
    with col1:
        # upload image
        im_file_buffer = st.file_uploader('Please upload a retinal fundus image.', type=["png", "jpg"])
        im = Image.open(im_file_buffer)
        im_array = np.array(im)
        if im is not None:
            st.image(
                im,
                caption=f"Image Size : {im_array.shape[0:2]}",
                use_column_width=True,
            )
        p = Predict(im)
        quality = p.predict_quality()
        diagnosis = p.predict_dr_level()

    with col2:
        st.subheader('`IMAGE QUALITY`')
        st.write(quality)
        if quality != 'Quality is `good` enough for the diagnosis of retinal diseases':
            st.write('Please upload another image.')

        if quality == 'Quality is `good` enough for the diagnosis of retinal diseases':
            st.subheader('`DIAGNOSIS`')
            st.write(diagnosis)