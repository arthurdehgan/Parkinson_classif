import pandas as pd
import numpy as np
import random
from path import Path as path
from sklearn.model_selection import LeavePOut, cross_val_score, StratifiedKFold, permutation_test_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier as RF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier
from numpy.random import permutation
from scipy.io import savemat, loadmat


def CleanDataset(dataset, columns_to_drop, threshold=30):
    '''Nettoie les données.'''
    dropped_columns = []
    dropped_subjects = []
    subject_list = dataset['CODE']
    for column in columns_to_drop:
        if column in dataset.keys():
            try:
                dataset = dataset.drop(column, 1)
                dropped_columns.append(column)
            except:
                print('there was a problem droping', column)
                None
            del columns_to_drop[columns_to_drop.index(column)]

    for thresh in reversed(range(1,threshold+1)):
        dropped_counter = {}
        for column in dataset:
            for i in range(len(dataset)):
                tested_value = dataset[column].iloc[i]
                try:
                    if type(tested_value) == str:
                        tested_value = tested_value.replace(',', '.')
                        a = float(tested_value)
                        dataset.set_value(dataset.index.values[i], column, a)
                except:
                    if column not in columns_to_drop:
                        columns_to_drop.append(column)
                    break
                if tested_value != tested_value or tested_value == float(-999):
                    if column not in dropped_counter.keys():
                        dropped_counter[column] = 1
                    else:
                        dropped_counter[column] += 1

        for column in dropped_counter.keys():
            if dropped_counter[column] >= thresh:
                if column not in columns_to_drop:
                    columns_to_drop.append(column)

        for column in columns_to_drop:
            if column in dataset.keys():
                try:
                    dataset = dataset.drop(column, 1)
                    dropped_columns.append(column)
                except:
                    print('there was a problem droping', column)
                    None
                del columns_to_drop[columns_to_drop.index(column)]

        subjects_to_drop = []
        bad_subjects = {}
        for index, row in dataset.iterrows():
            for element in row:
                if element != element:
                    dataset = dataset.drop(index)
                    break
                elif element == float(-999):
                    if subject_list[index] not in bad_subjects.keys():
                        bad_subjects[subject_list[index]] = [1, index]
                    else:
                        bad_subjects[subject_list[index]][0] += 1

        nb_bad_subjects = len(bad_subjects.keys())
        for subject in bad_subjects.keys():
            if bad_subjects[subject][0] >= thresh:
                subjects_to_drop.append(bad_subjects[subject][1])

        for index in subjects_to_drop:
            tested_subject = subject_list[index]
#             try:
#                 dataset = dataset.drop(index)
#                 subject_list = subject_list.drop(index)
#                 dropped_subjects.append(tested_subject)
#             except:
#                 print('there was a problem droping', tested_subject)
    return dataset, dropped_columns, dropped_subjects


def FindMinorClass(label0_index, label1_index):
    # Création d'outils de balancing des classes
    quantity_of_class = {'0': len(label0_index),
                         '1': len(label1_index)}


    nb_minority_class = min(quantity_of_class['0'], quantity_of_class['1'])

    for key in quantity_of_class.keys():
        if nb_minority_class == quantity_of_class[key]:
            minority_class = int(key)

    if minority_class == 0:
        major_class = 1
        minor_class_index = label0_index
        major_class_index = label1_index
    else:
        major_class = 0
        minor_class_index = label1_index
        major_class_index = label0_index

    minor_class_index = np.asarray(minor_class_index)
    return minority_class, major_class, minor_class_index, major_class_index


def CreateRandomBalancedDataset(dataset, minor_class_index, major_class_index, n_repet=1):
    random_major_indexes = []
    for i in range(n_repet):
        my_set = np.random.choice(major_class_index, len(minor_class_index), replace=False)
        random_major_indexes.append(my_set)
    return random_major_indexes


def SelectSubjects(dataset, conditions):
    # sélectionne les sujets qui correpondent à nos conditions
    cond = conditions
    if 4 in cond:
        cond.append(2)
        cond.append(1)
        del cond[cond.index(4)]
    for index, row in dataset.iterrows():
        look_at = row['Type de Conversion']
        if look_at not in cond:
            dataset = dataset.drop(index, 0)
    return dataset
