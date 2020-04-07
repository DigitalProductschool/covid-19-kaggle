# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json

import matplotlib.pyplot as plt
plt.style.use('ggplot')

root_path = '/home/dian-ai/Documents/Covid19/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()

meta_df.info()

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)

############
#Helper function
############

#file reader class

class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            try:
                self.paper_id = content['paper_id']
            except Exception as e:
                self.paper_id = ''
            self.abstract = []
            self.body_text= []
            
            # Abstract
            try:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            except Exception as e:
                pass
            # Body text
            
            try:
                for entry in content['body_text']:
                    self.body_text.append(entry['text'])
            except Exception as e:
                pass
            
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}:{self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)


#Helper function adds break after every words when character
# length reach to certain amount. This is for the interactive plot so 
#that hover tool fits the screen.

def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data



########################
    ##Load the Data into DataFrame
######################

#Using the helper functions, let's read in the articles into a 
#DataFrame that can be used easily:
    
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 100) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
    
    # also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 0: 
        # no abstract provided
        dict_['abstract_summary'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 100:
        # abstract provided is too long for plot, take first 300 words append with ...
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        dict_['abstract_summary'].append(summary + "...")
    else:
        # abstract is short enough
        summary = get_breaks(content.abstract, 40)
        dict_['abstract_summary'].append(summary)
        
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    
    try:
        # if more than one author
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            # more than 2 authors, may be problem when plotting, so take first 2 append with ...
            dict_['authors'].append(". ".join(authors[:2]) + "...")
        else:
            # authors will fit in plot
            dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # if only one author - or Null valie
        dict_['authors'].append(meta_data['authors'].values[0])
    
    # add the title information, add breaks when needed
    try:
        title = get_breaks(meta_data['title'].values[0], 40)
        dict_['title'].append(title)
    # if title was not provided
    except Exception as e:
        dict_['title'].append(meta_data['title'].values[0])
    
    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])
    
df_covid = pd.DataFrame(dict_, columns=['paper_id','abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])
df_covid.head()



dict_ = None

df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))
df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))
df_covid.head()




df_covid.info()

df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)

df_covid['abstract'].describe(include='all')
df_covid['body_text'].describe(include='all')
df_covid.info()

#It looks like we didn't have duplicates. Instead, it was articles without Abstracts.


# drop Null vales:
df_covid.dropna(inplace=True)
df_covid.info()



#removing punctuation from each text
import re

df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

#convert to lower text
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))
df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))

#labelling language
from langdetect import detect
df_covid['langue'] = df_covid['title'].apply(detect)

df_covid.to_csv(r'/home/dian-ai/Documents/Covid19/CORD-19-research-challenge/df_covid_lang_labels.csv', index=False)


#keeping only english language:
df_covid = df_covid.loc[df_covid['langue'] == 'en']
df_covid.to_csv(r'/home/dian-ai/Documents/Covid19/CORD-19-research-challenge/df_covid_en_only.csv', index=False)


text = df_covid.drop(["paper_id", "abstract", "abstract_word_count", "body_word_count", "authors", "title", "journal", "abstract_summary", "langue"], axis=1)
text_arr = text.stack().tolist()



df_covid.info()







#getting Doc2Vec

import gensim

def read_corpus(df, column, tokens_only=False):
    """
    Arguments
    ---------
        df: pd.DataFrame
        column: str 
            text column name
        tokens_only: bool
            wether to add tags or not
    """
    for i, line in enumerate(df[column]):
        
        tokens = gensim.parsing.preprocess_string(line)
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


import random
frac_of_articles = 0.01
train_df  = df_covid.sample(frac=frac_of_articles, random_state=42)
train_corpus = (list(read_corpus(text, 'body_text'))) 




# using distributed memory model
model = gensim.models.doc2vec.Doc2Vec(dm=1, vector_size=100, min_count=2, epochs=20, seed=42, workers=3)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)







#getiing word2vec

from gensim.models import Word2Vec
sentences = text_arr
model = Word2Vec(sentences, min_count =  1)

#save the model
words = list(model.wv.vocab)
model.wv.save_word2vec_format('model_bin')


from biobert_embedding.embedding import BiobertEmbedding
#from sentence_transformers import SentenceTransformer
model = BiobertEmbedding()
#model = SentenceTransformer('bert-base-nli-mean-tokens')