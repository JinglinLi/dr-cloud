![Tests + Pylint](https://github.com/JinglinLi/dr-cloud/workflows/Build%20and%20Test/badge.svg)

# dr-cloud

## GOAL : web app for diagnosis of diabetic retinopathy

## BACKGROUND :

## CONTENT :

data
- wrangling.py 
- eda.py

model
- init.py
- hp_diagnosis.py
- hp_imquality.py
- predict.py
- train_diagnosis.py
- train_quality.py

streamlit_app
- app.py
- home.py
- diagnosis_app.py

test
- test_predict.py

.github
- workflows/build.yml

config.py : path to adjust for local run vs cloud run
.pylintrc \
requirements.txt \
requirements_dev.txt \
setup.py

Note : datasets that wget from deepdr and kaggle should be kept in a separate folder to make local and cloud procedure consistant and simplify cloud setting up.

## RESULT

https://user-images.githubusercontent.com/82587457/134711624-c1e3208d-dd33-4bee-a2bc-c57a0173ca9e.mov



## USAGE

