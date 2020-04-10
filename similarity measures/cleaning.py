#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:05:46 2020

@author: dian-ai
"""



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# load the meta data from the CSV file using 4 columns (abstract, title, authors, publish_time),
df=pd.read_csv('home/dian-ai/Documents/Covid19/CORD-19-research-challenge/metadata.csv', usecols=['paper_id','title','abstract','authors','publish_time', 'body_text', 'journal'])
print (df.shape)
#drop duplicates
#df=df.drop_duplicates()
df = df.drop_duplicates(subset='abstract', keep="first")
#drop NANs 
df=df.dropna()
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()
#show 5 lines of the new dataframe
print (df.shape)
df.head()