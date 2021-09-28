![Tests + Pylint](https://github.com/JinglinLi/dr-cloud/workflows/Build%20and%20Test/badge.svg)

# dr-cloud

## GOAL : web app for diagnosis of diabetic retinopathy

## BACKGROUND :

## CONTENT :
`data/wrangling.py` -> prepare and save dataframes used for eda, training and evaluation
`data/eda.py` -> print and plot key information about dataset 
`dr_app/init.py`
`dr_app/hp_diagnosis.py`
`dr_app/hp_imquality.py`
`dr_app/predict.py`
`dr_app/train_diagnosis.py`
`dr_app/train_quality.py`
`test/test_predict.py`
`.github/workflows/build.yml`
`config.py` -> path to adjust for local run vs cloud run
`.pylintrc` 
`requirements.txt`
`requirements_dev.txt` 
`setup.py`

Note : datasets that wget from deepdr and kaggle should be kept in a separate folder to make local and cloud procedure consistant and simplify cloud setting up.

## RESULT

https://user-images.githubusercontent.com/82587457/134711624-c1e3208d-dd33-4bee-a2bc-c57a0173ca9e.mov



## USAGE

