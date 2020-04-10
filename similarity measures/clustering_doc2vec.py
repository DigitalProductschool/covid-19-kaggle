#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:32:50 2020

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


#create doc2vec
# Now when we have clean data lets transform this data into vectors. 
#Gensim's implementation of doc2vec needs objects of TaggedDocuments class of gensim.

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(list(df['processed_text']))]

model = Doc2Vec(documents, size=100, window=2, min_count=4, workers=4)


# By now we have a fully loaded doc2vec model of all the document vectors 
#we had in our data frame To print all the vectors

#appending all the vectors in a list for training
X=[]
for i in range(0,len(df)):
    X.append(model.docvecs[i])
    
    


# We will use the vectors created in the previous section to generate the 
#clusters using K-means clustering algorithm. Implementation of K-means
# available in sklearn, so I will be using that implementation.

#first reducing the dimension
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(X)

x_pca.shape


from sklearn.cluster import MiniBatchKMeans
import numpy as np
#create the kmeans object withe vectors created previously

k = 20
kmeans = MiniBatchKMeans(n_clusters= k)
y_pred = kmeans.fit_predict(x_pca)
#print all the labels


#craete a dictionary to get cluster data
clusters={0:[],1:[],2:[],3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[]}
for i in range(28971):
    clusters[y_pred[i]].append(' '.join(df.loc[df.index[i],'processed_text']))







from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=300)
X_embedded = tsne.fit_transform(X)


# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=1000, learning_rate=200)
# X_embedded = tsne.fit_transform(x_pca)


from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(18,15)})

# colors
palette = sns.hls_palette(20, l = 0.4, s=0.9)

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title("t-SNE Covid-19 Articles - Clustered(K-Means) - doc2vec body text")
# plt.savefig("plots/doc2vec_pca_k20_t-sne.png")
plt.show()












# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(pca_result[:,0], pca_result[:,1], hue=y_pred, legend='full', palette=palette)
plt.title("PCA Covid-19 Articles - Clustered (K-Means) - doc2vec body_text")
# plt.savefig("plots/pca_covid19_label_TFID.png")
plt.show()






# import hdbscan

# clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
# clusterer.fit(X)

# HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#     gen_min_span_tree=True, leaf_size=40, memory=Memory(cachedir=None),
#     metric='euclidean', min_cluster_size=15, min_samples=None, p=None)

# hdbscan.HDBSCAN(algorithm='best', allow_single_cluster=False, alpha=1.0,
#         approx_min_span_tree=True, cluster_selection_epsilon=0.0,
#         cluster_selection_method='eom', core_dist_n_jobs=4,
#         gen_min_span_tree=True, leaf_size=40,
#         match_reference_implementation=False, 
#         metric='euclidean', min_cluster_size=15, min_samples=None, p=None)
