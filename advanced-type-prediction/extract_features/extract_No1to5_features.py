#!/usr/bin/env python
# coding: utf-8

# In[1]:


from elasticsearch import Elasticsearch
from typing import Dict, List, Optional
from collections import Counter
import math,json


# In[2]:


def extract_features_1to5(DBpedia_map_type_entities:Dict,dp_type:str,query:str,k_list=[5,10,20,50,100],es = Elasticsearch(),index="dbpedia_entity_centric")-> Dict[str, float]:
    if not es.indices.exists(index):
        print(f'you need to index "dbpedia_entity_centric" dataset to elasticSearch')
        return None
    ECBM25_t_q={}
    for k in k_list:
        hits = es.search(index=index, q=query, _source=True, size=k )["hits"]["hits"]  
        score_list= [hit["_score"] for hit in hits]
        w_e_t=len(DBpedia_map_type_entities[dp_type])      
        ECBM25_t_q[f'ECBM25_t_q_{k}']=round(sum(score_list)/w_e_t,4)
    return ECBM25_t_q
  


# In[3]:

if __name__ == '__main__':
    filepath="../data/DBpedia_map_type_entities.json"
    with open(filepath, 'r',encoding='utf-8') as f:
        DBpedia_map_type_entities = json.load(f)
    print("------length:",len(DBpedia_map_type_entities))
    dp_type="dbo:Place"
    question="When was Bibi Andersson married to Per Ahlmark very green?"
    scores=extract_features_1to5(DBpedia_map_type_entities,dp_type,question)
    print("------scores:",scores)


# In[ ]:




