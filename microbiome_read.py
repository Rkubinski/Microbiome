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
def data_sep(newDF, inputcol):
	for i in inputcol:
		newDF[i]=seq[i]	

#we create new dataframes for healthy, UC and CD data
healthyData = DF_create()
CDData = DF_create()
UCData = DF_create()

#we separate our current data into these dataframes
data_sep(healthyData,healthy['dataID'])
data_sep(CDData,CD['dataID'])
data_sep(UCData,UC['dataID'])



'''
healthyData['AVG']=healthyData.mean(axis=1) 

avgandName= df[['#OTU ID', "AVG"]]
avgandName=avgandName[avgandName.AVG > 1]
df= avgandName

plt.bar(df['#OTU ID'],df["AVG"])
plt.ylim([0,1000])
plt.xticks(rotation=90)


plt.show()'''








