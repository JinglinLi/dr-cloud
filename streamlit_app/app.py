""" DIABETIC RETINOPATHY """ 

import streamlit as st
import home
import diagnosis_app
import project_report


PAGES = {
    "HOME": home,
    "DIAGNOSIS APP": diagnosis_app,
    "PROJECT REPORT": project_report
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