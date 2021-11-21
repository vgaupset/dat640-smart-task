#!/usr/bin/env python
# coding: utf-8

# In[4]:


import json


# In[12]:


def get_alltypes_test_questions(filename):
    
    """
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except:
        print(f'File \'{filename}\' not found.')
        return None
    types=[]
    i=0
    j=0
    for entry in data:
        i+=1
        if entry['question']==None:
            continue
        if entry['category']=='resource':
            j+=1
            #print(entry['type'])
            types+=entry['type']

    return set(types)


# In[13]:


# filename="../../smart-dataset/datasets/DBpedia/smarttask_dbpedia_test.json"
# all_testing_types=get_alltypes_test_questions(filename)
# len(all_testing_types)

