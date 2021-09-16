""" DIABETIC RETINOPATHY """ 

import streamlit as st
import streamlit_app.home
import streamlit_app.diagnosis_app


PAGES = {
    "HOME": streamlit_app.home,
    "DIAGNOSIS APP": streamlit_app.diagnosis_app
}

# radio version 
selection = st.sidebar.radio("GO TO", list(PAGES.keys()))
page = PAGES[selection]
page.app()

# # button version 
# if st.sidebar.button('HOME'):
#     page = home
# if st.sidebar.button('DIAGNOSTIC APP'):
#     page = diagnosis_app
# if st.sidebar.button('PROJECT REPORT'):
#     page = project_report
# page.app()