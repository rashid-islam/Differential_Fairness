# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:46:08 2019

@author: Rashid
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import differential_fairness as DF

#%%
# Computing Differential Fairness (DF) metric using smoothed EDF method (Equation-6) and DF bias amplification (Section V) 
# Source: James R. Foulds, Rashidul Islam, Kamrun Naher Keya, and Shimei Pan. An Intersectional Definition of Fairness. ArXiv preprint arXiv:1807.08362 [CS.LG], 2018
# Link: https://arxiv.org/pdf/1807.08362.pdf

#%%
# income predictions on "Census Income" dataset 
# race, gender & nationality selected as protected attributes
def load_census_data (path,check):
    # Output: features (X), targets (y) and protected attributes (S)
    column_names = ['age', 'workclass','fnlwgt','education','education_num',
                    'marital_status','occupation','relationship','race','gender',
                    'capital_gain','capital_loss','hours_per_week','nationality','target']
    input_data = (pd.read_csv(path,names=column_names,
                               na_values="?",sep=r'\s*,\s*',engine='python'))
    # sensitive attributes; we identify 'race','gender' and 'nationality' as sensitive attributes
    # race: white and non-white
    input_data['race'] = input_data['race'].map({'Black': 0,'White': 1, 'Asian-Pac-Islander': 0, 'Amer-Indian-Eskimo': 0, 'Other': 0})
    input_data['gender'] = (input_data['gender'] == 'Male').astype(int)
    input_data['nationality'] = (input_data['nationality'] == 'United-States').astype(int)
    
    protected_attribs = ['race', 'gender','nationality']
    S = (input_data.loc[:, protected_attribs])
   
    # targets; 1 when someone makes over 50k , otherwise 0
    if(check):
        # pre-splitted training dataset
        y = (input_data['target'] == '>50K').astype(int)    # target 1 when income>50K
    else:
        # pre-splitted test dataset
        y = (input_data['target'] == '>50K.').astype(int)    # target 1 when income>50K
    
    X = (input_data
         .drop(columns=['target'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))
    return X.values, y.values, S.values
# load the train dataset
X, y, S = load_census_data('data/adult.data',1)
# load the test dataset
test_X, test_y, test_S = load_census_data('data/adult.test',0)

#%% Measuring differential fairness (epsilon) of a labeled dataset

epsilon_data = DF.computeSmoothedEDF(test_S,test_y)
print(f"Differential fairness (epsilon) of the dataset: {epsilon_data: .3f}")

#%% Measuring differential fairness (epsilon) of the classifier on the same dataset

# train logistic regression as M(X) to replace the original label by the classifier's predictions
def logisticRegressionMx(X,test_X,y): 
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    test_X = scaler.transform(test_X)
    clfLR = LogisticRegression(C=1.0,random_state=0,solver='liblinear')
    clfLR.fit(X,y) 
    predictions = clfLR.predict(test_X)
    return predictions
# mechanism/classiifer's predicted labels on test data
classifier_predictions = logisticRegressionMx(X,test_X,y) 

# differential fairness (epsilon) of the classifier
epsilon_classifier = DF.computeSmoothedEDF(test_S,classifier_predictions)
print(f"Differential fairness (epsilon) of the classifier: {epsilon_classifier: .3f}")

#%% DF bias amplification measure

# When epsilon_data is the differential fairness of a labeled dataset and epsilon_classifier is the 
# differential fairness of a classifier measured on the same dataset, 
# (epsilon_classifier - epsilon_data)is a measure of the extent to which the classifier increases 
# the unfairness over the original data, a phenomenon that refers to as bias amplification

# Bias amplification is a more politically conservative fairness metric

bias_amplification = epsilon_classifier - epsilon_data
print(f"DF bias amplification by the classifier: {bias_amplification: .3f}")