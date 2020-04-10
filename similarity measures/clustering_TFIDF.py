#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:54:32 2020

@author: dian-ai
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json

import matplotlib.pyplot as plt
plt.style.use('ggplot')


df= pd.read_csv("text_processed.csv") 

df.index=range(0,len(df))


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=2**10)
X = vectorizer.fit_transform(df['processed_text'].values)




from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(X.toarray())

x_pca.shape




from sklearn.cluster import MiniBatchKMeans
import numpy as np
#create the kmeans object withe vectors created previously

k = 20
kmeans = MiniBatchKMeans(n_clusters= k)
y_pred = kmeans.fit_predict(x_pca)






#craete a dictionary to get cluster data
clusters={0:[],1:[],2:[],3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[]}
for i in range(28971):
    clusters[y_pred[i]].append(' '.join(df.loc[df.index[i],'processed_text']))
    
    
    
    
from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=300)
X_embedded = tsne.fit_transform(X.toarray())




from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(18,15)})

# colors
palette = sns.hls_palette(20, l = 0.4, s=0.9)

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title("t-SNE Covid-19 Articles - Clustered(K-Means) - TFIDF body text")

plt.show()
