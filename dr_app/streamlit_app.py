""" DIAGNOSIS APP """
import streamlit as st
from PIL import Image
import numpy as np
from predict import Predict


st.header('DIABETIC RETINOPATHY DIAGNOSTIC APP')

col1, col2 = st.columns(2)
with col1:
    # upload image
    uploaded_file = st.file_uploader('Please upload a retinal fundus image.', type=["png", "jpg"])
    if uploaded_file is not None:
        im = Image.open(uploaded_file)
        im_array = np.array(im)
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
