![Tests + Pylint](https://github.com/JinglinLi/dr-cloud/workflows/Build%20and%20Test/badge.svg)

# dr_app: a web app for diagnosing diabetic retinopathy

## BACKGROUND & GOAL:
Diabetic retinopathy is the leading cause of working-age blindness. Since symptom only shows in the later phase of the disease early detection is crucial. However, population screening requires a sufficient amount of eye experts.

The goal is to develop a web app that allows an assistant to upload an image and make a diagnosis. It can further control whether the quality of the image taken by the assistant is sufficient for diagnosis.

## DATA:
Source
- https://isbi.deepdr.org
- https://www.kaggle.com/c/aptos2019-blindness-detection

Image Qualty Dataset:
- 1600 images
- binary: 0 - insufficient quality; 1 - sufficient quality

Diagnosis Dataset:
- 5262 images
- 5 categories: 0 - No; 1 - Mild; 2 - Moderate; 3 - Severe; 4 - Proliferative

## PROCEDURE:
- data wrangling
- exploratory data analysis
- prilimnary training of image quality model
- prilimnary training of diagnosis model
- hyperparameter search image quality model : on google cloud platform
- hyperparameter search diagnosis model : on google cloud platform
- predict image quality and diagnosis using saved model
- develop and deploy streamlit web app
- automatic testing, contineous integration, packaging

## CONTENT :
`dr_app`: (current github repo folder) \
|-`config.py`: path variable for running locally/on cloud \
|-`data/wrangling.py`: prepare and save dataframes used for eda, training and evaluation \
|-`data/eda.py`: print and plot key information about dataset  \
|-`dr_app/__init__.py`: make import shorter \
|-`dr_app/train_diagnosis.py`: priliminary training  \
|-`dr_app/train_quality.py`: priliminary training \
|-`dr_app/hp_diagnosis.py`: hyperparameter tunning \
|-`dr_app/hp_imquality.py`: hyperparameter tunning \
|-`dr_app/predict.py`: make prediction \
|-`dr_app/streamlit_app.py`: web app \
|-`test/test_predict.py`: for automatic testing \
|-`requirements.txt`: requirementsfor app \
|-`requirements_dev.txt`: requirements for development \
|-`.pylintrc`: instructions for pylint \
|-`.github/workflows/build.yml`: for automated build and test \
|-`setup.py`: instructions for pip \
`dr_app_mnt`: (put data in a separate folder, not in this github folder, to facilitate set up on  cloud) \
|-`deepdr`: data from deepdr \
|-`kaggle`: data from kaggle

## RESULT
https://user-images.githubusercontent.com/82587457/134711624-c1e3208d-dd33-4bee-a2bc-c57a0173ca9e.mov
