#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import modules and set up logging.
from typing import Callable, Dict, List, Set, Tuple, Generator
import gensim.downloader as api
from gensim.models import Word2Vec
import gensim
import logging
import numpy as np
import os
import nltk
import re
nltk.download('averaged_perceptron_tagger')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[2]:


#to download the pretrained model of 'word2vec-google-news-300'
#make sure to use a 64 bit python
import struct
struct.calcsize("P") * 8
#!which python
#!which pip


# In[3]:


# Reduce logging level.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)


# In[6]:


#https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
#model_loaded.save('googleNews.d2v')
#model_loaded = gensim.models.keyedvectors.KeyedVectors.load('googleNews.d2v')


# In[7]:


# CONTENT_WORD_TYPE=["noun","adj","verb","adv"]
# POS_TAGS=["NN","JJ","VB","RB"]


# In[8]:


def preprocessDBOtype(dp_type:str)->str:
    dp_type=dp_type[len("dbo:"):]
    splitted_type=re.findall('[A-Z][a-z]*', dp_type)
    return " ".join(splitted_type).lower()
    #return " ".join(splitted_type)


# In[9]:


def parse_word_POStag(sentence:str)->Dict:
    """
    parse content words of a sentence with their POS tag
    argument:a sentence string
    return:dictionary,key is POS tag and value is the corresponding word
           a list of content words
    """
    tag_dict={}
    content_words=[]
    tokenized_sentence = nltk.word_tokenize(sentence)
    tagged_words=nltk.pos_tag(tokenized_sentence)
    for POS_tag in ["NN","JJ","VB","RB"]:
        for word,tag in tagged_words:
            if tag[:2]==POS_tag:
                content_words.append(word)
                temp=tag_dict.get(POS_tag,[])
                temp.append(word)
                tag_dict[POS_tag]=temp
                #print(tag_dict[POS_tag])
    return tag_dict,content_words


# In[21]:


def calc_pairwise_similarity(model_loaded,question_tagged:Dict,type_tagged:Dict)->List:
    """
    calculate pairwise similarity between 
    content words in the query and the type label
    """
    similarities=[]
    for POS_tag,words in type_tagged.items():
        if POS_tag in question_tagged.keys():
            for word1 in words:
                for word2 in question_tagged[POS_tag]:
                    try:
                        similarities.append(model_loaded.similarity(word1,word2))
                    except KeyError:
                        pass
    return similarities


       


# In[14]:


def extract_features_23to25(model_loaded,dp_type:str, question:str)->Tuple[float,float,float]:
    #get content words and parse dictonary
    question_tagged,question_content=parse_word_POStag(question)
    processed_type=preprocessDBOtype(dp_type)
    type_tagged,type_content=parse_word_POStag(processed_type)
    #get centroid
    question_centrality=model_loaded.rank_by_centrality(question_content, use_norm=True)
    type_centrality=model_loaded.rank_by_centrality(type_content, use_norm=True)
    question_centroid=question_centrality[0][1]
    type_centroid=type_centrality[0][1]
    #feature 23
    sim_aggr=round(model_loaded.similarity(question_centroid, type_centroid),4)
    
    pairwise_similarity=calc_pairwise_similarity(model_loaded,question_tagged,type_tagged)
    #feature 24,25
    sim_max=max(pairwise_similarity)
    sim_avg=round(sum(pairwise_similarity)/len(pairwise_similarity),4)
    
    return {"SIMAGGR_t_q":sim_aggr,"SIMMAX_t_q":sim_max,"SIMAVG_t_q":sim_avg}


# In[12]:

if __name__ == '__main__':
    model_loaded = api.load('word2vec-google-news-300')
    dp_type="dbo:GreatMusicFestival"
    question="When was Bibi Andersson married to Per Ahlmark very green?"
    print(extract_features_23to25(model_loaded,dp_type, question))

    #try another model
    # print(api.info('text8'))
    # text8_corpus = api.load('text8')
    # model = Word2Vec(text8_corpus) 
    # model=model.wv
    # dp_type="dbo:GreatMusicFestival"
    # question="When was Bibi Andersson married to Per Ahlmark very green?"
    # extract_features_23to25(model,dp_type, question)


# In[15]:





# In[22]:




