import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df3 = pd.read_csv("final_table.csv", sep=',') 

#########################################################
#transform non-numerical categorical variable into digit#
#########################################################
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df3['diagnosis'] = le.fit_transform(df3['diagnosis'])

x = input('do you wish to see what the column looks like? (Y/N)')

if x == 'Y':
    print(df3['diagnosis'])
    print('not that all the variables have been converted to a digit allowing the AI to consider them')

###############################
#now lets choose our variables#
###############################

#lets try all the counts 
X = df3.loc[:, 'IP8BSoli':'UncRumi6']

#with the diagnosis as our categorical variable
y = df3.loc[:,'diagnosis']

#######################################
#Separate the training and testing set#
#######################################
from sklearn import model_selection
results = model_selection.train_test_split(X, y, test_size = 0.2, shuffle = True)
X_train, X_test, y_train, y_test = results
