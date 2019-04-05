import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

def dispersion_dict(d):
	index_of_dispersion={}
	for k in dict.keys(d):
		mean = sum(d[k])/len(d[k])
		if mean>0:
			index_of_dispersion[k]=(np.std(d[k])*np.std(d[k]))/mean
		else: 
			continue
	return index_of_dispersion

def get_n_largest(d, n):
	
	return sorted([(value, key) for key, value in d.items()],
                  reverse=True)[:n]



#We create new dataframes separated by the diagnosis of the patient

healthyData = DF_create()
CDData = DF_create()
UCData = DF_create()
data_prep()						#Fill our new data frame with their corresponding data
data_dict=dict_prep()			#We transform our dataframes into a dict with OTU as key and the avg from each patient group as the value
								# ex: {(UBL293240: .23432, 1.325, 4.3223), .... }

######################################################################################
#				SELECTING THE MOST VARIANT bacteria                                  #
######################################################################################

dispersion=dispersion_dict(data_dict)	#we make a new dict with the key as the OTU and the value as the index of dispersion of the AVG values
most_variant=get_n_largest(dispersion,20)	#we now take our top 200 most variant bacteria


######################################################################################
#                                     GRAPHING                                       #
######################################################################################

x_labels=[]
healthy=[]
CD=[]
UC=[]
types=["Healthy","Crohns","Ulcerative Colitis"]


#Here we prepare some lists of our bacterial count data
for (v,k) in most_variant:
	ind = seq["#OTU ID"][seq["#OTU ID"]==k].index
	

	pd.set_option('display.max_colwidth', 300)
	genus = seq.taxonomy[ind].to_string()
	start = genus.rfind("__")
	genus=genus[start+2:]

	
	label= k#+" genus: "+genus
	
	x_labels.append(label)
	
	healthy.append(data_dict[k][0])
	CD.append(data_dict[k][1])
	UC.append(data_dict[k][2])
	

#Calculating some offset between our bars for our chart & plotting 
# i took this from stack exchange, no clue abt the logic
vals =[healthy,CD,UC]
n=len(vals)
_X=np.arange(len(x_labels))
width=.8
for i in range(n):
	if i==0:
		clr = 'g'
	if i==1:
		clr = 'y'
	if i==2:
		clr = 'r'
	plt.bar(_X - width/2. +i/float(n)*width,vals[i],width=width/float(n), color=clr ,align="edge")


plt.xticks(_X,x_labels,rotation=90)
plt.xlabel("Genera")
plt.ylabel("OTU count")
plt.title("Bacterial genera in different patient types")
plt.legend(types)
plt.tight_layout()
plt.ylim([0,2000])
plt.show()









