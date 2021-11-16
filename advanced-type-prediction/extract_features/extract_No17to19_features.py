#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from typing import Callable, Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[2]:



# In[3]:


FEATURES_17="LENGTH_t"
FEATURES_18="IDFSUM_t"
FEATURES_19="IDFAVG_t"


# In[4]:



# In[5]:


def extract_features_17to19(data,dp_type:str,query:str)-> Dict[str, float]:
    corpus=[item['questions'] for item in data]
    db_types=[item['type'] for item in data]
    #print(db_types)
    type_index=db_types.index(dp_type)
    #print(type_index)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    terms_corpus=vectorizer.get_feature_names()
    analyze = vectorizer.build_analyzer()
    query_terms=analyze(query)
    idf=[]
    for term in query_terms:
        if term in terms_corpus:
            idf.append(X[type_index,terms_corpus.index(term)])
        else:
            idf.append(0)
    return {
        #"LENGTH_t":len(corpus[type_index].split()),
        "LENGTH_t":len(dp_type),
        "IDFSUM_t":round(sum(idf),4),
        "IDFAVG_t":round(sum(idf)/len(idf),4)
    }  


# In[6]:

if __name__ == '__main__':
    filepath="../data/training_types.json"
    with open(filepath,encoding='utf-8') as json_file:
        data = json.load(json_file)
        
    dp_type='dbo:MusicalWork'
    dp_type="dbo:MusicFestival"
    question="When was Bibi Andersson music festival married to Per Ahlmark very green?"
    features=extract_features_17to19(data,dp_type,question)
    print(features)


# In[ ]:




