![Tests + Pylint](https://github.com/JinglinLi/dr-cloud/workflows/Build%20and%20Test/badge.svg)

## BACKGROUND & GOAL:
Diabetic retinopathy is the leading cause of working-age blindness. Since symptom only shows in the later phase of the disease early detection is crucial. However, population screening requires a sufficient amount of eye experts. 
The goal is to develop a web app that allows an assistant to upload an image and make a diagnosis. It can further control whether the quality of the image taken by the assistant is sufficient for diagnosis.

## Workflow: 
- data wrangling and EDA
- priliminary training
  - 1) a deep neural network for determining whether the quality of a fundus image is sufficient for diagnosis (binary) 
  - 2) a deep neural network for diagnosing diabetic retinopathy (5 stages)
- run hyperparameter tunng on google cloud platform (GCP)
- deploy a streamlit web app
- automatic testing, contineous integration, packaging

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
Note : datasets that wget from deepdr and kaggle should be kept in a separate folder to make local and cloud procedure consistant and simplify cloud setting up.

## RESULT:

https://user-images.githubusercontent.com/82587457/134711624-c1e3208d-dd33-4bee-a2bc-c57a0173ca9e.mov



## USAGE

