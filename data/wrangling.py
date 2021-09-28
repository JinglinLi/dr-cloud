""" 
prepare and save dataframe for eda, model training and evaluation : 

Data Source : 
deepdr : https://isbi.deepdr.org
kaggle : https://www.kaggle.com/c/aptos2019-blindness-detection/

merge two datasets : 
- deepdr dataset have : train, valid, test dataset with image quality and diagnosis label
- kaggle dataset have : train and test dataset with diagnosis label

goal : 
- use deepdr dataset for training quality model
- merged deepdr and kaggle dataset for training diagnosis model

therefore need to prepare following dataframes : -> save under ./output folder
training quality check model: (use only deepdr dataset)
    (use original train for train-valid spilit, use original valid as test --> so that can evaluate with test)
    q_traindf : columns = ['im_path', 'im_quality']
    q_testdf : columns = ['im_path', 'im_quality']
training diagnosis model: (merge deepdr and kaggle dataset)
    (merge deepdr train, valid, and keggle train --> train-test split)
    d_traindf : columns = ['im_path', 'diagnosis']
    d_testdf : columns = ['im_path', 'diagnosis']
if want to see kaggle score : 
    k_testdf : columns = ['id_code', 'diagnosis']
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import config

class DataWrangling:
    """ prepare data for model training """

    def generate_quality_df(self):
        """
        generate dataframe for training and evaluating image quality model : only deepdr dataset
        (use original train for train-valid spilit, use original valid as test --> so that can evaluate with test)
        save : ./output/q_traindf.csv and ./output/q_testdf.csv
        """

        # read csv containing labels corresponding to the images
        train_csv= f'{config.PATH_DISK}/data/deepdr/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv'
        test_csv = f'{config.PATH_DISK}/data/deepdr/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv'
        print(config.PATH_DISK)
        print(config.PATH_VM)
        print(train_csv)
        train = pd.read_csv(train_csv)
        test = pd.read_csv(test_csv)

        # generate dataframe with image path and overall quality lable
        traindf = pd.DataFrame()
        testdf = pd.DataFrame()

        traindf['im_path'] = train['image_path'].apply(lambda x : x.replace('\\', '/')).apply(lambda x : f'{config.PATH_DISK}/data/deepdr/regular_fundus_images/regular-fundus-training/Images'+x[24:]) # mac 
        testdf['im_path'] = test['image_path'].apply(lambda x : x.replace('\\', '/')).apply(lambda x : f'{config.PATH_DISK}/data/deepdr/regular_fundus_images/regular-fundus-validation/Images'+x[26:]) # mac 

        traindf['im_quality'] = train['Overall quality'].astype('str')
        testdf['im_quality'] = test['Overall quality'].astype('str')

        # save output
        traindf.to_csv(f'{config.PATH_VM}/data/output/q_traindf.csv')
        testdf.to_csv(f'{config.PATH_VM}/data/output/q_testdf.csv')

        #print(f'quality : total {traindf.shape[0] + testdf.shape[0]}, train {traindf.shape[0]}, test {testdf.shape[0]}')
        print('quality : total {}, train {}, test {}'.format(traindf.shape[0] + testdf.shape[0], traindf.shape[0], testdf.shape[0]))

    def generate_diagnosis_df_deepdr(self):
        """
        prepare dataframe for training diagnosis model : using deepdr data

        Note : this dataframe from deepdr dataset will be merged with the one 
        from kaggle dataset, in kaggle dataset train and valid images were not 
        separated, therefore here also merge train and valid, after mering with
        kaggle dataset train and valid will be splitted in model training part.
        """

        # read csv containing labels corresponding to the images
        train_csv= f'{config.PATH_DISK}/data/deepdr/regular_fundus_images/regular-fundus-training/regular-fundus-training.csv'
        valid_csv = f'{config.PATH_DISK}/data/deepdr/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv'
        train = pd.read_csv(train_csv)
        valid = pd.read_csv(valid_csv)

        # generate dataframe with image path and overall quality lable
        traindf = pd.DataFrame()
        validdf = pd.DataFrame()

        traindf['im_path'] = train['image_path'].apply(lambda x : x.replace('\\', '/')).apply(lambda x : f'{config.PATH_DISK}/data/deepdr/regular_fundus_images/regular-fundus-training/Images'+x[24:]) # mac 
        validdf['im_path'] = valid['image_path'].apply(lambda x : x.replace('\\', '/')).apply(lambda x : f'{config.PATH_DISK}/data/deepdr/regular_fundus_images/regular-fundus-validation/Images'+x[26:]) # mac 

        traindf['diagnosis'] = train['patient_DR_Level'].astype('str')
        validdf['diagnosis'] = valid['patient_DR_Level'].astype('str')

        return pd.concat([traindf, validdf])

    def generate_diagnosis_df_kaggle(self):
        """ prepare dataframe for training diagnosis model : using kaggle data"""
        # read csv containing labels corresponding to the images
        train_csv= f'{config.PATH_DISK}/data/kaggle/train.csv'
        test_csv = f'{config.PATH_DISK}/data/kaggle/test.csv'
        train = pd.read_csv(train_csv)
        test = pd.read_csv(test_csv) # only id no lable

        # generate dataframe with image path and overall quality lable
        traindf = pd.DataFrame()
        testdf = pd.DataFrame()

        traindf['im_path'] = train['id_code'].apply(lambda x : f'{config.PATH_DISK}/data/kaggle/train_images/'+x+'.png')
        testdf['im_path'] = test['id_code'].apply(lambda x : f'{config.PATH_DISK}/data/kaggle/test_images/'+x+'.png')
        
        traindf['diagnosis'] = train['diagnosis'].astype('str')
        testdf['diagnosis'] = ''

        # save kaggle diagnosis testdf
        testdf.to_csv(f'{config.PATH_VM}/data/output/d_testdf_kaggle.csv')
        return traindf

    def generate_diagnosis_df(self):
        """ combine diagnosis df from deepdr and kaggle
        Note : 
        1) concat deepdr, kaggle df
        2) train test split (shuffle)
        """
        deepdrdf = self.generate_diagnosis_df_deepdr()
        kaggledf = self.generate_diagnosis_df_kaggle()
        mergedf = pd.concat([deepdrdf, kaggledf]).sample(frac=1).reset_index(drop=True) # shuffle

        n = round(mergedf.shape[0] * 0.75)
        traindf = mergedf.iloc[:n]
        testdf = mergedf.iloc[n:]
        #print(f'diagnosis : total {mergedf.shape[0]}, train {traindf.shape[0]}, test {testdf.shape[0]}')
        print('diagnosis : total {}, train {}, test {}'.format(mergedf.shape[0], traindf.shape[0], testdf.shape[0]))

        traindf.to_csv(f'{config.PATH_VM}/data/output/d_traindf.csv')
        testdf.to_csv(f'{config.PATH_VM}/data/output/d_testdf.csv')

if __name__ == "__main__":
    dw = DataWrangling()
    dw.generate_quality_df() # generate and save dataframes for quality check
    dw.generate_diagnosis_df() # merge and save dataframes for diagnosis from deepdr and kabble dataset
