""" PROJECT REPORT """ 
from functools import total_ordering
from keras.backend import transpose
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def app():
    """project report"""
    st.header('**PROJECT REPORT : DIAGNOSING DIABETIC RETINOPATHY**')

    st.header('GOAL')
    st.write('develop a webapp for diagnosing diabetic retinopathy')

    st.header('DATA SOURCE')
    st.markdown('deepdr : https://isbi.deepdr.org')
    st.markdown('kaggle : https://www.kaggle.com/c/aptos2019-blindness-detection/')

    st.header('EDA')
    st.subheader('**Data Size**')
    col1, col2 = st.columns(2)
    with col1:
        st.write('`Image Quality Data`')
        st.bar_chart(pd.DataFrame(data=[1200*0.75, 1200*0.25, 400], columns=['n_data'], index=['train', 'valid', 'test']))
    with col2:
        st.write('`Diagnosis Data`')
        st.bar_chart(pd.DataFrame(data=[round(3946*0.75), round(3946*0.25), 1316], columns=['n_data'], index=['train', 'valid', 'test']))

    st.subheader('**Features**')
    st.write('png (kaggle) and jpg (deepdr) images of size 500-2000?')
    st.subheader('**Lables**')
    col1, col2 = st.columns(2)
    with col1:
        st.write('`Image Quality Labels:`')
        st.write('0 : bad quality')
        st.write('1 : good quality')
        # st.write('n training data')
        # st.bar_chart()
    with col2:
        st.write('`Diagnosis Labels : Diabetic Retinopathy Level`')
        st.write('0 : No DR')
        st.write('1 : Mild')
        st.write('2 : Moderate')
        st.write('3 : Severe')
        st.write('4 : Proliferative DR')
        
    st.header('MODEL')


    ########## penguin example
    # ## Add text
    # st.title('Penguins explorer')

    # ### st.write: army knife: pass text, data frames, and more
    # st.write('Demo app try out _plots_ and `dataframes` and *more*')
    # st.header('The Data')
    # st.image('lter_penguins.png')

    # df = pd.read_csv('penguins_pimped.csv')
    # df.dropna(inplace = True)
    # st.subheader('Very nice rendering of a data frame')
    # st.write(df.sample(20))

    # ## Add interactive elements
    # st.subheader('Selectbox')
    # species = st.selectbox('Which species do you want to see?', df['species'].unique())

    # df_species = df.loc[df['species'] == species,:]

    # st.subheader('Checkbox')
    # if st.checkbox('I want to see a small sample of the data frame'):
    #     df_small = df.loc[df['species'] == species,:].sample(3)
    #     df_small

    # # Plotting
    # st.header('Plots plots plots!')
    # st.subheader('Standard `matplotlib` and `seaborn`')
    # fig, ax = plt.subplots()
    # ax = sns.scatterplot(data = df, x = 'body_mass_g', y = 'flipper_length_mm', hue = 'species')
    # sns.despine()
    # st.pyplot(fig)

    # st.subheader('Streamlit figure')
    # st.write('Pass fitting data to display in e.g. a bar chart')
    # st.bar_chart(data = df.groupby('species')['island'].count())

