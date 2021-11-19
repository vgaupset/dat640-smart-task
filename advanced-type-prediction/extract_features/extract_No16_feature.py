#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re,os,json
from typing import Callable, Dict, List, Set, Tuple 


# In[ ]:



# In[ ]:


def extract_features_16(DBpedia_map_type_entities:Dict,dp_type:str)-> Dict[str, float]:
    if dp_type not in DBpedia_map_type_entities.keys():
        return {"ENTITIES_t": 0}
    return {"ENTITIES_t": len(DBpedia_map_type_entities[dp_type])}


# In[ ]:

if __name__ == '__main__':
    filepath="../data/DBpedia_map_type_entities.json"
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    #data.keys()
    DBO_type='dbo:Disease'
    DBO_type='dbo:NaturalEvent'
    print(extract_features_16(data,DBO_type))

