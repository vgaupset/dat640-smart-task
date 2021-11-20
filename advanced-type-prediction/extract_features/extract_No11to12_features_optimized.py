#!/usr/bin/env python
# coding: utf-8

# In[1]:


from elasticsearch import Elasticsearch
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import math,json


#get analysed term list
def produce_nGram_terms(nGram:int, sentence:str)->List:
    bigram_vectorizer = CountVectorizer(ngram_range=(nGram,nGram),token_pattern=r'\b\w+\b', min_df=1)
    analyze = bigram_vectorizer.build_analyzer()
    terms=analyze(sentence)
    return terms


# In[5]:


def get_query_term_count(term:str,query_terms:List)->int:
    query_term_freqs = Counter(query_terms)
    return query_term_freqs[term]


# In[7]:
def get_collection_paramter(es: Elasticsearch,field="abstract",index="dbpdiea_type_centric"):
    #these parameters are same for every doc
    tv = es.termvectors(index=index, doc_type="_doc", id="10", fields=field, term_statistics=True) 
    try:
        collection_len=tv["term_vectors"][field]['field_statistics']['sum_ttf']  
    except KeyError:
        return False
    #total number of entity in the collections
    doc_number=tv["term_vectors"][field]['field_statistics']['doc_count'] 
    #average document length
    avgdl=collection_len/doc_number
    return collection_len,doc_number,avgdl

def scorer_LM(es: Elasticsearch, doc_id:str,
              query_terms,index: str,
              collection_len:int,
              map_docID_docLength:Dict,
              field="abstract",smoothing_param=2000)->float:
 
    tv = es.termvectors(index=index, doc_type="_doc", id=doc_id, fields=field, term_statistics=True) 
    #print(tv)

    #abstract length/doc_length
    doc_len=map_docID_docLength[doc_id]
    
    doc_query_score=0
    for term in set(query_terms):
        try:
            term_count_collection=tv["term_vectors"][field]["terms"][term]["ttf"]
        except KeyError:
            term_count_collection=0

        try:
            count_t_d=tv["term_vectors"][field]["terms"][term]["term_freq"]
        except KeyError:
            count_t_d=0
        
        count_t_q=get_query_term_count(term,query_terms)
        P_t_givenC=term_count_collection/collection_len
       
        if count_t_d==0 and P_t_givenC==0:
            continue
      
        term_score=count_t_q*math.log((count_t_d+smoothing_param*P_t_givenC)/(doc_len+smoothing_param))
        doc_query_score+=term_score
            
        
    return doc_query_score


# In[9]:



    

def scorer_BM25(es: Elasticsearch, doc_id:str,
                query_terms:List,index: str,
                doc_number:int,avgdl:int,
                map_docID_docLength:Dict[str,int] ,
                field="abstract",k1=1.2,b=0.75)->float:
   
   
    tv = es.termvectors(index=index, doc_type="_doc", id=doc_id, fields=field, term_statistics=True) 

    #total number of entity in the collections

    #
    #doc_length
    doc_len=map_docID_docLength[doc_id]
    
    doc_query_score=0
    for term in set(query_terms):
        try:
            #number of documents containing term t
            
            n_t=tv["term_vectors"][field]["terms"][term]["doc_freq"]
            idf_t=math.log(doc_number/n_t)
        except KeyError:
            continue
            
        try:
            count_t_d=tv["term_vectors"][field]["terms"][term]["term_freq"]
        except KeyError:
            continue
            
        term_score=count_t_d*(1+k1)*idf_t/(count_t_d+k1*(1-b+b*doc_len/avgdl))
        doc_query_score+=term_score 

    return round(doc_query_score,4)

# In[10]:


def map_docID_DBOtype(es=Elasticsearch(),index="dbpdiea_type_centric")->Dict:
    if not es.indices.exists(index):
        print(f'Index: {index} not exist ')
        return None
    count = es.cat.count(index, params={"format": "json"})
    len_indexed=int(count[0]["count"])
    if len_indexed==0:
        print(f'No items exist in {index}')
        return None
  
    return {es.get(index=index, id=str(i))['_source']["type"]:str(i) for i in range(len_indexed)}


# In[8]:


def extract_features_11to12(dp_type:str, query:str,docID_DBOtype_dict:Dict,
                            map_docID_docLength:Dict,
                            collection_len:int,
                            doc_number:int,
                            avgdl:float,
                            es:Elasticsearch,index="dbpdiea_type_centric",
                            smoothing_param=2000,k1=1.2,b=0.75
                            )->Dict:
    if not es.indices.exists(index):
        print(f'you need to index "dbpdiea_type_centric" dataset to elasticSearch')
        return None
    doc_id=docID_DBOtype_dict.get(dp_type,"not found")
    if doc_id=="not found" :
        #print("type not found in elasticSearch")
        return {"TCBM25_t_q": 0,
                "TCLM_t_q": 0}
    field="abstract"
    query_terms=produce_nGram_terms(1,query)
    return {"TCBM25_t_q": round(scorer_BM25(es, doc_id,query_terms,index,doc_number,avgdl,map_docID_docLength,field,k1,b),4),
            "TCLM_t_q": round(scorer_LM(es, doc_id,query_terms,index,collection_len,map_docID_docLength,field,smoothing_param),4)}


# In[11]:

if __name__ == '__main__':
    es = Elasticsearch()
    es.info()
    count = es.cat.count("dbpdiea_type_centric", params={"format": "json"})
    print("--count",int(count[0]["count"]))


    filepath="../data/ElasticSearch_map_type_docID.json"
    with open(filepath, 'r',encoding='utf-8') as f:
        docID_DBOtype_dict = json.load(f)
    print("------length:",len(docID_DBOtype_dict))

    filepath="../data/DBpedia_map_docID_docLength.json"
    with open(filepath, 'r',encoding='utf-8') as f:
         map_docID_docLength = json.load(f)
    print("------length:",len(map_docID_docLength))
   
      
    index="dbpdiea_type_centric"
    collection_len,doc_number,avgdl=get_collection_paramter(es,field="abstract",index="dbpdiea_type_centric")
    dp_type="dbo:Place"
    #dp_type="dbo:TimePeriod"
    question="When was Bibi Andersson married to Per Ahlmark very green?"
    print(extract_features_11to12(dp_type, question,docID_DBOtype_dict,map_docID_docLength,collection_len,doc_number,avgdl,es))
    
    #test with small dataset, to check the implementation is correct or not
    # query="t3"
    # index="toy_index"
    # field="body"
    # doc_id="d1"
    # doc = es.get(index=index, id=doc_id)
    # #print(doc)
    # print("BM25:",scorer_BM25(es, doc_id,query,index,field))
    # print("LM:",scorer_LM(es, doc_id,query,index,field))


