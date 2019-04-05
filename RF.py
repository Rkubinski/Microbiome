import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from math import isnan
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


def preprocess():
	taxonom_data= pd.read_csv("taxonomic_profiles.tsv", sep='\t', index_col=False)
	taxonom_data=taxonom_data.rename(columns={'#OTU ID':"Patient ID"})
	taxonom_data=taxonom_data.drop("taxonomy",axis=1)
	patient_IDs=taxonom_data.columns.values.tolist()
	patient_IDs.pop(0)

	# now we get the disease state. 
	meta_data= pd.read_csv("hmp2_metadata.csv", low_memory=False)
	meta_data=meta_data[["External ID","diagnosis"]].head(178)
	metaDICT = dict(zip(meta_data['External ID'].values.tolist(),meta_data['diagnosis'].values.tolist()))
	index_map = {v: i for i, v in enumerate(patient_IDs)}
	metaDICT=sorted(metaDICT.items(), key=lambda pair: index_map[pair[0]])
	metaDICT=[x[1] for x in metaDICT]
	metaDICT.insert(0,"diagnosis")

	taxonom_data.loc[len(taxonom_data), :] = metaDICT


	CD=['Patient ID']
	UC=['Patient ID']
	diseaseOnly=['Patient ID']

	for col in taxonom_data.columns:

		diag= taxonom_data[col].iloc[982]
	
		if diag=="nonIBD":
			UC.append(col)
			CD.append(col)
			taxonom_data[col].iloc[982]=0
		
		elif (diag =="UC"):
			taxonom_data[col].iloc[982] =1 
			diseaseOnly.append(col)
			UC.append(col)
		
		elif (diag=="CD"):
				
			diseaseOnly.append(col)				
			taxonom_data[col].iloc[982] =2 #when we compare healthy to diseases, healthy gets 0 and disease gets 1
			
			CD.append(col)
	
	diseased= taxonom_data.filter(diseaseOnly,axis=1)
	CD_Data = taxonom_data.filter(CD, axis=1)
	UC_Data = taxonom_data.filter(UC, axis=1)
	
	
	diseased.to_csv("CDvsUC_preprocessed_data.csv",sep="\t", index=False)
	#we needed labels 1 and 2 to differentatiate Crohn's and UC so we could do CD vs UC
	#now, we can switch back to 1 and 0, so we can differentiate between IBD and healthy (generally sick vs healthy)
	for col in taxonom_data.columns:

		diag= taxonom_data[col].iloc[982]
		if diag==2:
			taxonom_data[col].iloc[982]=1
	taxonom_data.to_csv("total_preprocessed_data.csv",sep="\t", index=False)
	CD_Data.to_csv("CD_preprocessed_data.csv",sep="\t", index=False)
	UC_Data.to_csv("UC_preprocessed_data.csv",sep="\t", index=False)
	
	return taxonom_data['Patient ID'].drop(982).tolist()

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
	feature_list=OTUs
	feature_importances = feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
	feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

	# List of features sorted from most to least important
	sorted_importances = [importance[1] for importance in feature_importances]
	sorted_features = [importance[0] for importance in feature_importances]

	tx = pd.read_csv("taxonomic_profiles.tsv",sep='\t')
	pd.set_option('display.max_colwidth', 300)	#make sure we get the full genera name		
	#print("3 most important bacterial genera used for decision making: ")
	names=[]
	values=[]
	for i in range(0,10):
		#row=tx[tx["#OTU ID"]==sorted_features[i]]
		#idx=row.index
		#genus=tx.iloc[idx]['taxonomy'].to_string()
		#start = genus.rfind("__")
		#genus=genus[start+2:]
		names.append(sorted_features[i])
		values.append(sorted_importances[i])
	

	
	

	#Getting our most important values -> these are >.001 contribution
	important_feature_names=[imp[0] for imp in feature_importances if imp[1]>0]
	#important_feature_names = [feature[0] for feature in feature_importances[0:10]]

	# Find the columns of the most important features
	important_indices = [feature_list.index(feature) for feature in important_feature_names]
	important_train_features = X_train[:, important_indices]
	important_test_features = X_test[:, important_indices]

	return names,values,important_train_features,important_test_features



OTUs=preprocess()
files={"IBD":"total_preprocessed_data.csv","Crohn's Disease":"CD_preprocessed_data.csv","Ulcerative Colitis":"UC_preprocessed_data.csv","Crohns vs UC": "CDvsUC_preprocessed_data.csv"}
genera=[]
numericImportance=[]
titles=["IBD vs Healthy","Crohns vs Healthy","Ulcerative Colitis vs Healthy","Crohns vs Ulcerative Colitis"]


for key in files.keys():
	print("Running microbiome-based prediction for disease: "+key)

	X,y=generateDataset(files[key])
	#train test split
	X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.2 ,random_state=42)	
	#Random forest
	clf = RandomForestClassifier(n_estimators=1000,random_state=2,max_features=982)
	clf.fit(X_train,y_train)
	print("Accuracy of Random Forest Classifier: "+str(clf.score(X_test, y_test)))
	importances=list(clf.feature_importances_)

	names,values,important_train_features,important_test_features=reduceFeatures(importances)
	genera.append(names)			#we will use this to graph later
	numericImportance.append(values)
	max_features= len(important_train_features[0])

	clf = RandomForestClassifier(n_estimators=1000,random_state=2,max_features=max_features)
	clf.fit(important_train_features,y_train)

	print("Accuracy After Feature Reduction: "+str(clf.score(important_test_features, y_test)))

fig, axes = plt.subplots(nrows=2, ncols=2)
counter=0
for row in axes:
	for col in row:
		col.bar(genera[counter],numericImportance[counter])
		counter+=1 
		
		



fig.text(0.5, 0.0, 'Genera', ha='center')
fig.text(0.0, 0.5, 'Importances', va='center', rotation='vertical')
fig.suptitle("OTU vs Importance")
counter=0
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=90)

    ax.set_title(titles[counter])
    counter+=1

plt.tight_layout()
plt.show()


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
'''