import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("taxonomic_profiles.tsv",sep='\t')
df= df.head(178)





df ['AVG']=df.mean(axis=1) 

avgandName= df[['#OTU ID', "AVG"]]
avgandName=avgandName[avgandName.AVG > 1]
df= avgandName

plt.bar(df['#OTU ID'],df["AVG"])
plt.ylim([0,1000])
plt.xticks(rotation=90)


plt.show()



