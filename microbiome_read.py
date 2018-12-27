import pandas as pd
import numpy as np

########################################################################################
# Make the table containing all the 16s data of interest and the info of each partient #
########################################################################################

#load and process the metadata
metadata = pd.read_csv("hmp2_metadata.csv", low_memory=False) 
df = metadata[["External ID","diagnosis"]]	#extract only columns we want
df = df.head(178)	#only take the 16S rRNA assay
df.columns= ['dataID','diagnosis']	#rename the column headers

#load the sequencing data
seq =pd.read_csv("taxonomic_profiles.tsv",sep='\t')



#categorizing patient data
healthy=df.loc[df["diagnosis"] == "nonIBD"]
UC=df.loc[df["diagnosis"] == "UC"]
CD=df.loc[df["diagnosis"] == "CD"]

#defining useful functions 
def DF_create ():
	temp =pd.DataFrame(data=seq["#OTU ID"])
	temp.columns=["OTU"]
	return temp
def df_prep(newDF, inputcol):
	for i in inputcol:
		newDF[i]=seq[i]	
	newDF['AVG']=newDF.mean(axis=1)

def data_prep():
	
	df_prep(healthyData,healthy['dataID'])
	df_prep(CDData,CD['dataID'])
	df_prep(UCData,UC['dataID'])

def dict_prep():
	d={}
	keys=healthyData['OTU']
	for s in keys:
		for j in healthyData[healthyData['OTU']==s].index:
			d[s]=[healthyData['AVG'][j],CDData['AVG'][j],UCData['AVG'][j]] 
	return d
def var_dict(d):
	variance={}
	for k in dict.keys(d):
		variance[k]=np.std(d[k])*np.std(d[k])
	return variance
def get_n_largest(dictionary, n):
    return sorted([(value, key) for key, value in dictionary.items()],
                  reverse=True)[:n]




#we create new dataframes separated by the diagnosis of the patient

healthyData = DF_create()
CDData = DF_create()
UCData = DF_create()

data_prep()
data_dict=dict_prep()
variance=var_dict(data_dict)

#we now have our most variant bacteria 
most_variant=get_n_largest(variance,200)






'''


avgandName= df[['#OTU ID', "AVG"]]
avgandName=avgandName[avgandName.AVG > 1]
df= avgandName

plt.bar(df['#OTU ID'],df["AVG"])
plt.ylim([0,1000])
plt.xticks(rotation=90)


plt.show()'''








