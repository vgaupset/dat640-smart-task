#!/usr/bin/env python
# coding: utf-8

# In[1]:


from elasticsearch import Elasticsearch
import string,re,sys
from typing import Dict, List, Optional
from collections import Counter
import math,json
sys.path.insert(0, '../util')
from helper_function import preprocess





def extract_features_1to5(DBpedia_map_type_entities:Dict,dp_type:str,
                          query:str,es:Elasticsearch,
                          k_list=[5,10,20,50,100],index="dbpedia_entity_centric"
                          )-> Dict[str, float]:
    """calculate Entity centric 
    """
    if not es.indices.exists(index):
        print(f'you need to index "dbpedia_entity_centric" dataset to elasticSearch')
        return None
    if dp_type not in DBpedia_map_type_entities.keys():
      
        return {f'ECBM25_t_q_{k}':0 for k in k_list}
    
    ECBM25_t_q={}
    hits = es.search(index=index, q=query, _source=True, size=max(k_list) )["hits"]["hits"]  
    score_list= [hit["_score"] for hit in hits]
    w_e_t=len(DBpedia_map_type_entities[dp_type])      
    for k in k_list:
        ECBM25_t_q[f'ECBM25_t_q_{k}']=round(sum(score_list[0:k])/w_e_t,4)
    return ECBM25_t_q
  


# In[3]:

if __name__ == '__main__':
    
    
    filepath="../data/DBpedia_map_type_entities.json"
    with open(filepath, 'r',encoding='utf-8') as f:
        DBpedia_map_type_entities = json.load(f)
    print("------length:",len(DBpedia_map_type_entities))
    dp_type="dbo:Place"
    #dp_type='dbo:NaturalEvent'
    question="When was Bibi Andersson married to Per Ahlmark very green?"
    #question="Who is {famous for} of {writers} of {To the Christian Nobility of the German Nation} ?"
    es= Elasticsearch()
    question=preprocess(question)
    print("------question:",question)
    scores=extract_features_1to5(DBpedia_map_type_entities,dp_type,question,es)
    print("------scores:",scores)


# In[ ]:




