import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ML_preprocess import getOTU 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

#prepping our file
df = pd.read_csv("UC_preprocessed_data.csv",sep='\t')
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

#train test split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.2 ,random_state=42)
feature_list=getOTU()





#Random forest
clf = RandomForestClassifier(n_estimators=1000,random_state=2,max_features=982).fit(X_train,y_train)

print("Accuracy of Random Forest Classifier: "+str(clf.score(X_test, y_test)))

importances=list(clf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)


# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
print(feature_importances)
'''
# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)
x_values = list(range(len(importances)))
important_feature_names = [feature[0] for feature in feature_importances[0:5]]
print(important_feature_names)
# Find the columns of the most important features
important_indices = [feature_list.index(feature) for feature in important_feature_names]
# Create training and testing sets with only the important features
important_train_features = X_train[:, important_indices]
#important_train_labels=y_train[:,important_indices]
important_test_features = X_test[:, important_indices]
#important_test_labels=y_test[:, important_indices]

# Sanity check on operations
#print('Important train features shape:', important_train_features.shape)
#print('Important test features shape:', important_test_features.shape)

#print(len(important_train_features),len(y_train))

clf = RandomForestClassifier(n_estimators=1000,random_state=2).fit(important_train_features, y_train)


# Make predictions on test data
predictions = clf.predict(important_test_features)
errors = abs(predictions - y_test)
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / len(y_test))
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('New Accuracy:', round(accuracy, 2), '%.')

#number_of_features = len(feature_names) #the length of the input vector
#plt.barh(range(number_of_features), clf.feature_importances_, align='center')
#plt.yticks(np.arange(number_of_features), feature_names)
#plt.xlabel('Feature importance')
#plt.ylabel('Feature name')
#plt.show()

#estimator=clf.estimators_[5]

####################################################################################
#                        VISUALIZATION OF RANDOM FOREST 						   #
####################################################################################

#for tree visualization
#class_names=["IBD","nonIBD"]
#feature_list=getOTU()

#from sklearn.tree import export_graphviz

#export_graphviz(estimator, out_file='tree.dot', rounded=True,proportion=False,precision=2,filled=True,class_names=class_names,feature_names=feature_names)

#from subprocess import call
#call(['dot','-Tpng','tree.dot','-o','tree.png','Gdpi=600'])

#clf3 = GradientBoostingClassifier(n_estimators=1000, learning_rate=1,
#max_depth=10, random_state=0, min_samples_split=5).fit(X_train, y_train)
#print ("Accuracy of Gradient Boosting Classifier: "+str(clf3.score(X_test,y_test)))
'''