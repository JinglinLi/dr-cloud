"""
EDA : print and plot essential information about the data

"""

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dr_app.config as config

def print_dataset_size():
    """print the size of imquality and diagnosis dataset"""
    q_traindf = pd.read_csv(f'{config.PATH_VM}/data/output/q_traindf.csv')
    q_testdf = pd.read_csv(f'{config.PATH_VM}/data/output/q_testdf.csv')
    q_total = q_traindf.shape[0] + q_testdf.shape[0]
    q_train = round(q_traindf.shape[0]*0.8)
    q_valid = round(q_traindf.shape[0]*0.2)
    q_test = q_testdf.shape[0]
    print('    ')
    print('--------the size of image quality dataset-------')
    print('image quality dataset : total {}, train {}, valid{}, test {}'.format(q_total, q_train, q_valid, q_test))

    d_traindf = pd.read_csv(f'{config.PATH_VM}/data/output/d_traindf.csv')
    d_testdf = pd.read_csv(f'{config.PATH_VM}/data/output/d_testdf.csv')

    d_total = d_traindf.shape[0] + d_testdf.shape[0]
    d_train = round(d_traindf.shape[0]*0.8)
    d_valid = round(d_traindf.shape[0]*0.2)
    d_test = d_testdf.shape[0]
    print('    ')
    print('--------the size of diagnosis dataset-------')
    print('diagnosis : total {}, train {}, valid{}, test {}'.format(d_total, d_train, d_valid, d_test))
    print('    ')

def print_train_class_counts():
    """print class counts of imquality and diagnosis dataset"""
    q_traindf = pd.read_csv(f'{config.PATH_VM}/data/output/q_traindf.csv')
    d_traindf = pd.read_csv(f'{config.PATH_VM}/data/output/d_traindf.csv')

    print('    ')
    print('--------image quality dataset class counts-------')
    print(q_traindf['im_quality'].value_counts().sort_index())

    print('    ')
    print('--------diagnosis dataset class counts-------')
    print(d_traindf['diagnosis'].value_counts().sort_index())

def plot_diagnosis_class_images():
    """plot example image of each diagnosis category"""
    d_traindf = pd.read_csv(f'{config.PATH_VM}/data/output/d_traindf.csv')
    for i in [0, 1, 2, 3, 4]:
        d = d_traindf[d_traindf['diagnosis'] == i].sample(1)
        d_file = d['im_path'].values[0]
        image = mpimg.imread(d_file)
        plt.subplot(1,5,i+1)
        plt.title(f'category label : {i}')
        plt.imshow(image,cmap='gray',interpolation='none')
    plt.show()

def plot_imquality_class_images():
    """plot example image of each imquality category"""
    q_traindf = pd.read_csv(f'{config.PATH_VM}/data/output/q_traindf.csv')
    for i in [0, 1]:
        q = q_traindf[q_traindf['im_quality'] == i].sample(1)
        q_file = q['im_path'].values[0]
        image = mpimg.imread(q_file)
        plt.subplot(1,2,i+1)
        plt.title(f'category label : {i}')
        plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    print_dataset_size()
    print_train_class_counts()
    plot_diagnosis_class_images()
    plot_imquality_class_images()
