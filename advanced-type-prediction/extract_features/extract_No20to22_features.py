#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import re
from typing import  Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
nltk.download('averaged_perceptron_tagger')


# In[2]:


# In[3]:


def jaccard_similarity(setA:Set, setB:Set)->float:
    union_sets=setA.union(setB)
    inter_sets=setA.intersection(setB)
    return len(inter_sets)/len(union_sets)


# In[4]:


def produce_nGram_terms(nGram:int, sentence:str)->set:
    bigram_vectorizer = CountVectorizer(ngram_range=(nGram,nGram),token_pattern=r'\b\w+\b', min_df=1)
    analyze = bigram_vectorizer.build_analyzer()
    terms=analyze(sentence)
    return set(terms)


# In[5]:


def split_DBOtype(dp_type:str)->set:
    dp_type=dp_type[len("dbo:"):]
    splitted_type=re.findall('[A-Z][a-z]*', dp_type)
    joined_type= " ".join(splitted_type).lower()
    return joined_type


# In[7]:


def get_nouns(sentence:str)->set:
    """
    get all nouns from string
    return set of these nouns
    """  
    is_noun = lambda pos: pos[:2] == 'NN'    
    tokenized = nltk.word_tokenize(sentence)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
    return set(nouns) 


# In[8]:


def extract_features_20to22(dp_type:str, question:str)->Tuple[float,float,float]:
    question_1_gram=produce_nGram_terms(1,question)
    question_2_gram=produce_nGram_terms(2,question)
    question_nouns=get_nouns(question)
    dp_type_processed=split_DBOtype(dp_type)
 
    type_1_gram=produce_nGram_terms(1,dp_type_processed)
    type_2_gram=produce_nGram_terms(2,dp_type_processed)
    type_nouns=get_nouns(dp_type_processed)

    
    try:
        feature20=jaccard_similarity(question_1_gram,type_1_gram)
    except:
        feature20=0
    try:
        feature21=jaccard_similarity(question_2_gram,type_2_gram)
    except:
        feature21=0
    try:
        feature22=jaccard_similarity(question_nouns,type_nouns)
    except:
        feature22=0
    return {"JTERMS1_t_q": round(feature20,4), "JTERMS2_t_q": round(feature21,4),"JNOUNS_t_q":round(feature22,4)}
           


# In[9]:

if __name__ == '__main__':
    dp_type="dbo:MusicFestival"
    dp_type="dbo:Single"
    question="When was Bibi Andersson music festival married to Per Ahlmark very green?"
    question="What is it?"
    print(extract_features_20to22(dp_type, question))

