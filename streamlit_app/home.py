""" HOME PAGE """ 

import streamlit as st


"""home"""
def app():
    st.header('`DIABETIC RETINOPATHY`')
    st.write('Diabetic retinopathy is a complication of diabetes, caused by high blood sugar levels damaging the back of the eye (retina). It can cause blindness if left undiagnosed and untreated. (https://www.nhs.uk/conditions/diabetic-retinopathy/)')

    st.header('`Most common cause of blindness 25-65 ages`')

    st.subheader('Without ML : screening is difficult <- limited specialized doctors')
    st.subheader('With ML : screening is possible <- ML-model + assistants + specialized doctors')

    st.header('`Stop blindness before it is too late`')
