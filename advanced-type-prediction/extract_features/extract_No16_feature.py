#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re,os,json
from typing import Callable, Dict, List, Set, Tuple 


# In[ ]:


FEATURES_16="LENGTH_t"


# In[ ]:


def extract_features_16(data,dp_type:str)-> Dict[str, float]:
    return {"LENGTH_t": len(data[dp_type])}


# In[ ]:

if __name__ == '__main__':
    filepath="../data/DBpedia_map_type_entities.json"
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    #data.keys()
    DBO_type='dbo:Disease'
    print(extract_features_16(data,DBO_type))

