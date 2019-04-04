import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

#prepping our file
df = pd.read_csv("total_preprocessed_data.csv",sep='\t')
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
Xs=scale(X)
y=np.array(labels)

#########################################################################################################################
# DATA PREPROCESS COMPLETE#
#########################################################################################################################


kmeans_model = KMeans(n_clusters=2,random_state=1)
kmeans_model.fit(Xs)
lbls= kmeans_model.labels_

pca =PCA(2)
plot_columns=pca.fit_transform(Xs)
plt.scatter(x=plot_columns[:,0],y=plot_columns[:,1],c=lbls)
plt.show()

