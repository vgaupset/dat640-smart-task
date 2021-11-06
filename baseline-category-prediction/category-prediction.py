from typing import Any, Dict, List, Union
import json
import os

from elasticsearch import Elasticsearch
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier

def load_categories(path: str) -> List[dict]:
    """Load categories with doc ids for smarttask

    Args:
        path: path to json file to load.

    Returns:
        Tuple with list of doc ids and list of categories
    """
    doc_ids = []
    categories = []
    f = open(path,)
    for object in json.load(f):
        if object['question']:
            doc_ids.append(object['id'])
            categories.append(object['category'])
    f.close()
    return doc_ids, categories



def get_feature_vectors(es: Elasticsearch, index_name: str, doc_ids: List[str], vocabulary: List[str]):
    feature_vectors = []
    feature_vector_len = len(vocabulary)
    for doc_id in doc_ids:
        feature_vector = [0] * feature_vector_len
        termvectors = es.termvectors(index=index_name, id=doc_id, fields="question")['term_vectors']
        if 'question' in termvectors and 'terms' in termvectors['question']:
            terms = termvectors['question']['terms']
            for term,term_info in terms.items():
                if term in vocabulary:
                    feature_vector[vocabulary.index(term)] = term_info['term_freq']
        else:
            print("could not find terms")
            print(termvectors)
        feature_vectors.append(feature_vector)
    return feature_vectors


def get_vocabulary(es: Elasticsearch, index_name: str, doc_ids: List[str]):
    vocabulary = []
    for doc_id in doc_ids:
        termvectors = es.termvectors(index=index_name, id=doc_id, fields="question", term_statistics=True)['term_vectors']
        if 'question' in termvectors and 'terms' in termvectors['question']:
            for key, info in termvectors['question']['terms'].items():
                if info['doc_freq'] > 3 and info['ttf'] > 3:
                    vocabulary.append(key)
        else:
            print(doc_id)
    return list(set(vocabulary))

def train(X,y):
    # text_clf = Pipeline([('vect', CountVectorizer()),
    #                     ('tfidf', TfidfTransformer()),
    #                     ('clf', MultinomialNB()),
    # ])
    # pipe = make_pipeline(Normalizer(),MultinomialNB())
    return RandomForestClassifier().fit(X, y)

def load_vocabulary(train_doc_ids):
    if os.path.exists("./vocabulary.json"):
        f = open("./vocabulary.json",)
        vocabulary = json.load(f)
        f.close()
    else:
        vocabulary = get_vocabulary(es, "smart-task-train", train_doc_ids)
        f = open("./vocabulary.json","w+")
        json.dump(vocabulary,f)
        f.close()
    return vocabulary

def evaluate(actual:List[str], predicted:List[str]):
    length = len(actual)
    no_correct = 0
    for i, actual_category in enumerate(actual):
        if actual_category == predicted[i]:
            no_correct += 1
    return no_correct/length


if __name__ == '__main__':
    es = Elasticsearch()
    es.info()
    train_doc_ids, train_categories = load_categories("./smart-dataset/datasets/DBpedia/smarttask_dbpedia_train.json")
    test_doc_ids, test_categories = load_categories("./smart-dataset/datasets/DBpedia/smarttask_dbpedia_test.json")

    vocabulary = load_vocabulary(train_doc_ids)
    train_feature_vectors = get_feature_vectors(es, "smart-task-train", train_doc_ids, vocabulary)
    test_feature_vectors = get_feature_vectors(es, "smart-task-test", test_doc_ids, vocabulary)
    
    print("Training...")
    classifier = train(train_feature_vectors, train_categories)

    print("Applying model on test data...")
    predicted_categories = classifier.predict(test_feature_vectors)

    print("Evaluating")
    precision = evaluate(test_categories, predicted_categories)

    print(f"Precision:\t{precision}")
