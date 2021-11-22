#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json,sys
from typing import Callable, Dict, List, Set, Tuple
import csv
import pickle
import numpy as np
import gensim.downloader as api
import gensim
import datetime
from elasticsearch import Elasticsearch
sys.path.insert(1, 'extract_features')
sys.path.insert(1, 'util')
from helper_function import preprocess
from sklearn.ensemble import RandomForestRegressor
from extract_No17to19_features import get_analyze,extract_features_17to19
from extract_No20to22_features import extract_features_20to22
from extract_No23to25_features_optimized import extract_features_23to25


# In[27]:


# Reduce logging level.
import logging
logging.basicConfig(level=logging.ERROR)
np.seterr(all="ignore")


# In[3]:


#load model



# In[5]:


def extract_features_type_label(
    question:str,
    dp_type: str,
    training_map_type_questions:Dict,
    model_loaded:gensim.models.keyedvectors.KeyedVectors,
    #es: Elasticsearch,
    analyze,X,terms_corpus,
    add_extra_features:bool=False
) -> List[float]:
    """Extracts features of a query and document pair.

        Args:
            query: string.
            dp_type: DBO type.
            es: Elasticsearch object instance.

        Returns:
            List of extracted feature values in a fixed order.
    """  
    features_17to19=extract_features_17to19(analyze,X,terms_corpus,training_map_type_questions,dp_type,question)
    feature_vect=list(features_17to19.values())
    
    features_20to22=extract_features_20to22(dp_type,question)
    feature_vect.extend(list(features_20to22.values()))
    
    features_23to25=extract_features_23to25(model_loaded,dp_type, question, mode="Euclidean")
    feature_vect.extend(list(features_23to25.values()))

    if add_extra_features:
        features_23to25_variant=extract_features_23to25(model_loaded,dp_type, question, mode="similarities")
        feature_vect.extend(list(features_23to25_variant.values()))
    
    return feature_vect


# In[6]:


def load_training_data(filepath)-> Tuple[List[List[float]], List[int]]:
    print("----the training data is loading from:",filepath)
    X_train=[]
    y_train=[]
    file = open(filepath)
    csvreader = csv.reader(file)

    for i,line in enumerate(csvreader):
        if i%100000==0:
            print("------",i)
        if i%2==0:
            X_train.append([float(item) for item in line])
        else:
            y_train.append(int(line[0]))
            
    file.close()

    return X_train,y_train


# In[7]:


class PointWiseLTRModel:
    def __init__(self) -> None:
        """Instantiates LTR model with an instance of scikit-learn regressor.
        """
        self.regressor = RandomForestRegressor()
        self.model=None

    def _train(self, X: List[List[float]], y: List[float]) -> None:
        """Trains an LTR model.

        Args:
            X: Features of training instances.
            y: Relevance assessments of training instances.
        """
        assert self.regressor is not None
        self.model = self.regressor.fit(X, y)

    def rank(
        self, ft: List[List[float]], baseline_result: List[str]
    ) -> List[Tuple[str, int]]:
        """Predicts relevance labels and rank documents for a given query.

        Args:
            ft: A list of feature vectors for query-document pairs.
            doc_ids: A list of document ids.
        Returns:
            List of tuples, each consisting of document ID and predicted
                relevance label.
        """
        assert self.model is not None
        rel_labels = self.model.predict(ft)
        sort_indices = np.argsort(rel_labels)[::-1]

        results = []
        for i in sort_indices:
            results.append((baseline_result[i], rel_labels[i]))
        return results



# In[25]:


def get_rankings(
    ltr: PointWiseLTRModel,
    training_map_type_questions:Dict[str,str],
    model_loaded:gensim.models.keyedvectors.KeyedVectors,
    filepath_baseline:str,
    filepath_testing:str,
    result_path:str,
    es: Elasticsearch
) -> Dict[str, List[str]]:
    """Generate rankings for each of the test query IDs.

    Args:
        ltr: A trained PointWiseLTRModel instance.
        query_ids: List of query IDs.
        es: Elasticsearch object instance.
        index: Name of relevant index on the running Elasticsearch service.
        rerank: Boolean flag indicating whether the first-pass retrieval
            results should be reranked using the LTR model.

    Returns:
        A dictionary of rankings for each test query ID.
    """
    analyze,X,terms_corpus=get_analyze(training_map_type_questions)
    
    with open(filepath_testing, 'r') as f:
        smarttask_dbpedia_test = json.load(f)
        
    with open(filepath_baseline, 'r') as f:
        map_testID_ranktype = json.load(f)
    
    test_rankings={}
    count=0
    print("-----------------",datetime.datetime.now())
    with open(result_path, 'w', newline='') as csvfile:
        print("---------------result_path:",result_path)
        writer = csv.writer(csvfile)

        for entry in smarttask_dbpedia_test:
            

            if entry['question']==None:
                continue
                
            question_processed=preprocess(entry['question']) 
            
            if question_processed.isspace():
                test_rankings[entry['id']] ="NIL"
                writer.writerow([entry['id']])
                writer.writerow(["NIL"])
                continue
            

            if entry['category']=='resource':

                if count%30==0:
                    print("-----------------",datetime.datetime.now())
                    print(f'{count} questions has been processed')
                count+=1

                try:
                    baseline_result=map_testID_ranktype[entry['id']]
                except KeyError:
                    question_processed=preprocess(entry['question'])                         
                    hits = es.search(
                        index="dbpdiea_type_centric", q=question_processed, _source=True, size=30
                    )["hits"]["hits"]
                    baseline_result= [hit['_source']["type"] for hit in hits]
                
                #print(len(baseline_result),baseline_result)
                # Rerank the first-pass result set using the LTR model.
                features_list=[]
                for DBOtype in baseline_result:    
                   

                    feature=extract_features_type_label(question_processed,
                                                                DBOtype,
                                                                training_map_type_questions,
                                                                model_loaded,
                                                                #es,
                                                                analyze,X,terms_corpus)     

                    features_list.append(feature)
                
                #print(len(features_list),features_list)
                
                if features_list==[]:
                    print(entry)
                    print(question_processed)
                    test_rankings[entry['id']] ="NIL"
                    writer.writerow([entry['id']])
                    writer.writerow(["NIL"])
                    continue
                try:
                    rtest=ltr.rank(features_list,baseline_result) 
                    #print(len(rtest),rtest)
                    reranked_type=[item[0] for item in rtest]
                    test_rankings[entry['id']] =reranked_type
                except:
                    print(entry)
                    print(question_processed)
                    print(features_list)
                    raise
                writer.writerow([entry['id']])
                writer.writerow(reranked_type)
            
    return test_rankings


# In[10]:

if __name__ == '__main__':
    filepath="data/training_types.json"
    with open(filepath,encoding='utf-8') as json_file:
        training_map_type_questions = json.load(json_file)
        
        
    try:
        model_loaded = gensim.models.keyedvectors.KeyedVectors.load('googleNews.d2v')
    except:
        model_loaded = api.load('word2vec-google-news-300')
        model_loaded.save('googleNews.d2v')
        model_loaded = gensim.models.keyedvectors.KeyedVectors.load('googleNews.d2v')

    
    try:
        filename = 'data/finalized_model.sav'
        ltr=PointWiseLTRModel()
        ltr.model = pickle.load(open(filename, 'rb'))
    except:
        filepath="data/for_training_type_label.csv"
        X_train,y_train=load_training_data(filepath)
        ltr=PointWiseLTRModel()
        ltr._train(X_train,y_train)
        print("finish training")
        # save the model to disk
        filename = 'data/finalized_model.sav'
        pickle.dump(ltr.model, open(filename, 'wb'))
        print("trained model has been saved")
        # load the model from disk
        ltr.model = pickle.load(open(filename, 'rb'))
        print("trained model has been loaded")
        
        
        

# In[4]:





# In[ ]:



# In[28]:





# In[ ]:





# In[ ]:



