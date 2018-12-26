import pandas as pd
import numpy as np

########################################################################################
# Make the table containing all the 16s data of interest and the info of each partient #
########################################################################################

#load the metadata
metadata = pd.read_csv("hmp2_metadata.csv", low_memory=False) 
df = metadata[["External ID","diagnosis"]]
df = df.head(178)


#extracting the healthy patient data
healthy=df.loc[df["diagnosis"] == "nonIBD"]

#matching healthy patients with their stats
healthyData = pd.read_csv("taxonomic_profiles.tsv/206719_taxonomic_profile.tsv",sep='\t')
healthyData = healthyData[]

