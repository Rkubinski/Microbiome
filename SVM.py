from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import confusion_matrix

#prepping our file
df = pd.read_csv("CD_preprocessed_data.csv",sep='\t')
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
X=np.array(xx)

y=np.array(labels)



"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)






clf = svm.NuSVC(decision_function_shape = 'ovo',gamma='auto')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Prediction accuracy: " +str(accuracy))
"""

X = StandardScaler().fit_transform(X)
pca = sklearnPCA(.95) #2-dimensional PCA

transformed = pca.fit_transform(X)
"""
principaldf=pd.DataFrame(data = transformed
             , columns = ['principal component 1', 'principal component 2'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['nonIBD', 'CD']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
"""