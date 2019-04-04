import numpy as np
import pandas as pd

df = pd.read_csv("CD_preprocessed_data.csv",delimiter='\t')
#OTU= df['Patient ID'].head(982).values.tolist()

OTU_data = df.drop('Patient ID',axis=1)
#gets the first patient's bacteria count

#have to get rid of the diagnosis row
OTU_data.drop(OTU_data.index[982],inplace=True)
OTU_data=OTU_data['206646'].astype('float64')
OTU_data=OTU_data.tolist()





def NN(bacteria,weights,b):
	sum=0.0
	
	for pos in range(len(bacteria)):

		sum=sum+(bacteria[pos]*weights[pos])
	sum=sum+b

	print(sum)
	return (sigmoid(sum))		# what kind of function to use here hmhmhm
def sigmoid(x):
	
	return 1/(1-np.exp(-x))

def generate_weights():
	
	return np.random.randn(len(OTU_data))

def generate_bias():
	return np.random.randn()

def cost(b):
	return (b-1)**2

print(NN(OTU_data,generate_weights(),generate_bias()))