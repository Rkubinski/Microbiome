import pandas as pd
import numpy as np
from math import isnan



taxonom_data= pd.read_csv("taxonomic_profiles.tsv", sep='\t', index_col=False)



taxonom_data=taxonom_data.rename(columns={'#OTU ID':"Patient ID"})
taxonom_data=taxonom_data.drop("taxonomy",axis=1)


patient_IDs=taxonom_data.columns.values.tolist()
patient_IDs.pop(0)



# now we get the disease state. 

meta_data= pd.read_csv("hmp2_metadata.csv", low_memory=False)
meta_data=meta_data[["External ID","diagnosis"]].head(178)
#CE = metadata[["CRP (mg/L)","ESR (mm/hr)"]].head(178)

metaDICT = dict(zip(meta_data['External ID'].values.tolist(),meta_data['diagnosis'].values.tolist()))
index_map = {v: i for i, v in enumerate(patient_IDs)}
metaDICT=sorted(metaDICT.items(), key=lambda pair: index_map[pair[0]])
metaDICT=[x[1] for x in metaDICT]
metaDICT.insert(0,"diagnosis")

taxonom_data.loc[len(taxonom_data), :] = metaDICT





CD=['Patient ID']
UC=['Patient ID']

for col in taxonom_data.columns:
	diag= taxonom_data[col].iloc[982]
	
	if diag=="nonIBD":
		UC.append(col)
		CD.append(col)
		taxonom_data[col].iloc[982]=0
		
	elif (diag =="UC"):
		taxonom_data[col].iloc[982] =1 
		
		UC.append(col)
		
	elif (diag =="nonIBD" or diag=="CD"):
		taxonom_data[col].iloc[982] =1 
		
		CD.append(col)
		
CD_Data = taxonom_data.filter(CD, axis=1)
UC_Data = taxonom_data.filter(UC, axis=1)
taxonom_data.to_csv("total_preprocessed_data.csv",sep="\t", index=False)
CD_Data.to_csv("CD_preprocessed_data.csv",sep="\t", index=False)
UC_Data.to_csv("UC_preprocessed_data.csv",sep="\t", index=False)

def getOTU():
	return taxonom_data['Patient ID'].drop(982).tolist()
