import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ML_preprocess import getOTU 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def generateDataset(filename):
	#prepping our file
	df = pd.read_csv(filename,sep='\t')
	df = df.drop('Patient ID',axis=1)

	#we recover our labels
	labels= df.values[-1].tolist()
	#converting to ints because pandas gives floats
	labels=[round(l) for l in labels]


	#get rid of the actual labels in our training set
	df.drop(df.index[982],inplace=True)
	#loop through the columns and save them as list of lists
	cols=df.columns.tolist()
	xx=[]
	for c in cols:
		xx.append(df[c].tolist())

	#creating our 2d matrix
	X=np.array(xx)
	#we scale our OTUs 
	X=scale(X)
	#we add in our lables
	y=np.array(labels)

	return X,y

def reduceFeatures(classifier):
	#we format this into tuples 
	feature_list=getOTU()
	feature_importances = feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
	feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

	# List of features sorted from most to least important
	sorted_importances = [importance[1] for importance in feature_importances]
	sorted_features = [importance[0] for importance in feature_importances]

	#Getting our most important values -> these are >.001 contribution
	important_feature_names=[imp[0] for imp in feature_importances if imp[1]>0]
	#important_feature_names = [feature[0] for feature in feature_importances[0:10]]

	# Find the columns of the most important features
	important_indices = [feature_list.index(feature) for feature in important_feature_names]
	important_train_features = X_train[:, important_indices]
	important_test_features = X_test[:, important_indices]

	return important_train_features,important_test_features


files={"IBD":"total_preprocessed.csv","Crohn's Disease":"CD_preprocessed.csv","Ulcerative Colitis":"UC_preprocessed.csv"}

for key in files.keys():
	print("Running microbiome-based prediction for disease: "+key)


	#train test split
	X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.2 ,random_state=42)	
	#Random forest
	clf = RandomForestClassifier(n_estimators=1000,random_state=2,max_features=982)
	clf.fit(X_train,y_train)
	print("Accuracy of Random Forest Classifier: "+str(clf.score(X_test, y_test)))
	importances=list(clf.feature_importances_)

	important_train_features,important_test_features=reduceFeatures(importances)
	max_features= len(important_train_features[0])

	clf = RandomForestClassifier(n_estimators=1000,random_state=2,max_features=max_features)
	clf.fit(important_train_features,y_train)



	print("Accuracy After Feature Reduction: "+str(clf.score(important_test_features, y_test)))


#include this if you want the genus name
#Printing the genus of the top 10 most important OTUs
#tx = pd.read_csv("taxonomic_profiles.tsv",sep='\t')
#pd.set_option('display.max_colwidth', 300)	#make sure we get the full genera name		
'''
row=tx[tx["#OTU ID"]==sorted_features[i]]
idx=row.index
genus=tx.iloc[idx]['taxonomy'].to_string()
start = genus.rfind("__")
genus=genus[start+2:]
print(genus)
