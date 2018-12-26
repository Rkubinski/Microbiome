import pandas as pd

df = pd.read_csv("taxonomic_profiles.tsv/taxonomic_profiles.tsv",sep='\t')
df= df.head(178)





df ['AVG']=df.mean(axis=1) 

avgandName= df[['#OTU ID', "AVG"]]

print(avgandName)

