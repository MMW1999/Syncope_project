## based on physionet_challenge_utility_script.py by Bjørn-Jostein
import os
import numpy as np, sys,os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ecg_plot
from scipy.io import loadmat
import wfdb
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
#from keras.utils import plot_model


# from Physionet_challenge_utility_script
def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

# function to plot ecg data from file
def plot_ecg(path, SampleRate, title):
    ecg_data = load_challenge_data(path)
    ecg_plot.plot(ecg_data[0]/1000, sample_rate=SampleRate, title=title)
    ecg_plot.show()

# function for loading key data from .mat files 
# from Physionet_challenge_utility_script
def import_key_data(path):
    gender=[]
    age=[]
    labels=[]
    ecg_filenames=[]
    for subdir, dirs, files in sorted(os.walk(path)):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".mat"):
                data, header_data = load_challenge_data(filepath)
                labels.append(header_data[15][5:-1])
                ecg_filenames.append(filepath)
                gender.append(header_data[14][6:-1])
                age.append(header_data[13][6:-1])
    return gender, age, labels, ecg_filenames

# from Physionet_challenge_utility_script
def clean_up_gender_data(gender):
  gender = np.asarray(gender)
  gender[np.where(gender == "Male")] = 0
  gender[np.where(gender == "male")] = 0
  gender[np.where(gender == "M")] = 0
  gender[np.where(gender == "Female")] = 1
  gender[np.where(gender == "female")] = 1
  gender[np.where(gender == "F")] = 1
  gender[np.where(gender == "NaN")] = 2
  gender[np.where(gender == "Unknown")] = 2
  np.unique(gender)
  gender = gender.astype(int)
  return gender

# from Physionet_challenge_utility_script
def clean_up_age_data(age):
    age = np.asarray(age)
    age[np.where(age == "NaN")] = -1
    age[np.where(age == "Male")] = -1
    np.unique(age)
    age = age.astype(int)
    return age

# from Physionet_challenge_utility_script
def import_gender_and_age(age, gender):
    gender_binary = clean_up_gender_data(gender)
    age_clean = clean_up_age_data(age)
    print("gender data shape: {}".format(gender_binary.shape[0]))
    print("age data shape: {}".format(age_clean.shape[0]))
    return age_clean, gender_binary

def make_undefined_class(labels, df_unscored):
    df_labels = pd.DataFrame(labels, columns=['SNOMED labels'])
    for i in range(len(df_unscored.iloc[0:,1])):
        df_labels.replace(to_replace=str(df_unscored.iloc[i,1]), inplace=True ,value="undefined class", regex=True)
    
    #codes also not classified and not in df_unscored
    codes_to_remove=['106068003', '17366009', '233892002', '251166008', '251187003', '251205003', 
                     '251223006', '365413008', '418818005', '426183003', '50799005', '55827005', 
                     '5609005', '61277005', '61721007', '67751000119106', '733534002']
    for i in range(len(codes_to_remove)):
        df_labels.replace(to_replace=codes_to_remove[i], inplace=True ,value="undefined class", regex=True)


    #equivalent classes
    codes_to_replace=['713427006','284470004','427172004']
    replace_with = ['59118001','63593006','17338001']

    for i in range(len(codes_to_replace)):
        df_labels.replace(to_replace=codes_to_replace[i], inplace=True ,value=replace_with[i], regex=True)
   
    return df_labels


def MLB_encode(df_labels):
    mlb = MultiLabelBinarizer()
    y=mlb.fit_transform(df_labels[0].str.split(pat=','))

    print("The classes we will look at are encoded as SNOMED CT codes:")
    print(mlb.classes_)

    # deleting the column undefined class, which contains all classes that 
    # were not taken into account for classification purposes
    #y = np.delete(y, -1, axis=1)
    y = y[:, :-2]
    print("classes: {}".format(y.shape[1]))

    return y, mlb.classes_[0:-2]

#def plot_classes(SNOMED_classes, classification_SNOMEDmapping, y):
def plot_classes(SNOMED_codes, corresponding_diagnoses, y):
    # updating classes variable with diagnoses corresponding to SNOMED-CT codes orignally in classes
    for j in range(len(SNOMED_codes)):
        for i in range(len(corresponding_diagnoses.iloc[:,1])):
            if (str(corresponding_diagnoses.iloc[:,1][i]) == SNOMED_codes[j]):
                SNOMED_codes[j] = corresponding_diagnoses.iloc[:,0][i]
    plt.figure(figsize=(20,80))
    plt.bar(x=SNOMED_codes, height=y.sum(axis=0))
    plt.title("Distribution of Diagnosis", color = "black", fontsize = 30)
    plt.tick_params(axis="both", colors = "black")
    plt.xlabel("Diagnosis", color = "black")
    plt.ylabel("Count", color = "black")
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize = 20)
    #plt.savefig("fordeling.png")
    plt.show()

def get_labels_for_all_combinations(y):
    y_all_combinations = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    return y_all_combinations

def split_data(labels, y_all_combo):
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=33).split(labels,y_all_combo))
    print("Training split: {}".format(len(folds[0][0])))
    print("Validation split: {}".format(len(folds[0][1])))
    return folds

def plot_all_folds(folds,y,onehot_enc):
    X_axis_labels=onehot_enc
    plt.figure(figsize=(20,100))
    h=1
    for i in range(len(folds)):
        plt.subplot(10,2,h)
        plt.subplots_adjust(hspace=1.0)
        plt.bar(x= X_axis_labels, height=y[folds[i][0]].sum(axis=0))
        plt.title("Distribution of Diagnosis - Training set - Fold {}".format(i+1) ,fontsize="20", color = "black")
        plt.tick_params(axis="both", colors = "black")
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize = 10)
        #plt.xlabel("Diagnosis", color = "white")
        plt.ylabel("Count", color = "black")
        h=h+1
        plt.subplot(10,2,h)
        plt.subplots_adjust(hspace=1.0)
        plt.bar(x= X_axis_labels, height=y[folds[i][1]].sum(axis=0))
        plt.title("Distribution of Diagnosis - Validation set - Fold {}".format(i+1) ,fontsize="20", color = "black")
        plt.tick_params(axis="both", colors = "black")
        #plt.xlabel("Diagnosis", color = "white")
        plt.ylabel("Count", color = "black")
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize = 10)
        h=h+1

def generate_validation_data(ecg_filenames, y,test_order_array):
    y_train_gridsearch=y[test_order_array]
    ecg_filenames_train_gridsearch=ecg_filenames[test_order_array]

    ecg_train_timeseries=[]
    for names in ecg_filenames_train_gridsearch:
        data, header_data = load_challenge_data(names)
        data = pad_sequences(data, maxlen=5000, truncating='post',padding="post")
        ecg_train_timeseries.append(data)
    X_train_gridsearch = np.asarray(ecg_train_timeseries)

    X_train_gridsearch = X_train_gridsearch.reshape(ecg_filenames_train_gridsearch.shape[0],5000,12)

    return X_train_gridsearch, y_train_gridsearch

def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        class_labels = np.unique(y_true[:, i])
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_true[:, i])
        weights[i] = class_weights
    return weights

def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A

def plot_normalized_conf_matrix(y_pred, ecg_filenames, y, val_fold, threshold, snomedclasses, snomedabbr):
    conf_m = compute_modified_confusion_matrix(generate_validation_data(ecg_filenames,y,val_fold)[1], (y_pred>threshold)*1)
    conf_m = np.nan_to_num(conf_m)
    #min_max_scaler = preprocessing.MinMaxScaler()
    #conf_m_scaled = min_max_scaler.fit_transform(conf_m)
    normalizer = preprocessing.Normalizer(norm="l1")
    conf_m_scaled = normalizer.fit_transform(conf_m)
    df_norm_col = pd.DataFrame(conf_m_scaled)
    df_norm_col.columns = snomedabbr
    df_norm_col.index = snomedabbr
    df_norm_col.index.name = 'Actual'
    df_norm_col.columns.name = 'Predicted'
    #df_norm_col=(df_cm-df_cm.mean())/df_cm.std()
    plt.figure(figsize = (12,10))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_norm_col, cmap="rocket_r", annot=True,cbar=False, annot_kws={"size": 10},fmt=".2f")# 