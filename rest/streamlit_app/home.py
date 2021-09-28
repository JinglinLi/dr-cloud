""" HOME PAGE """ 

import streamlit as st
import config

"""home"""
def app():
    st.header('`DIABETIC RETINOPATHY`')
    st.write('- retina desease caused by diatetes')
    st.write('- blood vessal wall damage due to high blood sugar level')
    st.write('- can cause blindness if left undiagnosed and untreated')

    st.header('`Most common cause of blindness 25-65 ages`')

    st.image(f'{config.PATH_VM}/streamlit_app/motivation.png')
    
    st.header('`Stop blindness before it is too late`')
