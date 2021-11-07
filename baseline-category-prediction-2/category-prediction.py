from typing import Any, Dict, List, Union
import json
import os

from sklearn.ensemble import RandomForestClassifier

QUESTION_WORDS = [
    "when",
    "who",
    "do",
    "what",
    "was",
    "which",
    "did",
    "how",
    "in",
    "is",
    "where",
    "name"
]
def load_categories(path: str) -> List[dict]:
    """Load categories with doc ids for smarttask

    Args:
        path: path to json file to load.

    Returns:
        Tuple with list of doc ids and list of categories
    """
    doc_ids = []
    categories = []
    questions = []
    f = open(path,)
    for object in json.load(f):
        if object['question']:
            doc_ids.append(object['id'])
            categories.append(object['category'])
            questions.append(object['question'])
    f.close()
    return doc_ids, categories, questions


def preprocess(doc: str):
    special_chars = [",", ".", ":", ";", "?", "!",
                     "\"", "\\", "_", "<", ">", "(", ")", "/", "@", "{", "}"]
    for char in special_chars:
        doc = doc.replace(char, ' ')
    return [word for word in doc.lower().split()]


def preprocess_multiple(docs: List[str]):
    """Preprocesses multiple texts to prepare them for feature extraction.

    Args:
        docs: List of strings, each consisting of the unprocessed contents
            of question.

    Returns:
        List of strings, each comprising the corresponding preprocessed
            text.
    """
    return list(map(preprocess, docs))


def get_vocabulary(docs: List[List[str]]):
    vocabulary = []
    for doc in docs:
        for term in doc:
            vocabulary.append(term)
    unique_terms = list(set(vocabulary))
    pop_terms = []
    for term in unique_terms:
        if vocabulary.count(term) < 3:
            pop_terms.append(term)
    for term in pop_terms:
        unique_terms.pop(unique_terms.index(term))
    return unique_terms

def load_vocabulary(docs):
    if os.path.exists("./vocabulary.json"):
        f = open("./vocabulary.json",)
        vocabulary = json.load(f)
        f.close()
    else:
        vocabulary = get_vocabulary(docs)
        f = open("./vocabulary.json", "w+")
        json.dump(vocabulary, f)
        f.close()
    return vocabulary


def get_feature_vector(doc: List[str], vocabulary):
    feature_vector = [0] * len(vocabulary) 
    for term in doc:
        if term in vocabulary:
            feature_vector[vocabulary.index(term)] = doc.count(term)
    if doc and len(doc) > 0 and doc[0] in QUESTION_WORDS:
        feature_vector.append(QUESTION_WORDS.index(doc[0]))
    else:
        feature_vector.append(0)
    return feature_vector


def get_feature_vectors(docs, vocabulary: List[str]):
    return [get_feature_vector(doc, vocabulary) for doc in docs]



def train(X, y):
    return RandomForestClassifier().fit(X, y)


def evaluate(actual: List[str], predicted: List[str]):
    length = len(actual)
    no_correct = 0
    for i, actual_category in enumerate(actual):
        if actual_category == predicted[i]:
            no_correct += 1
    return no_correct/length


if __name__ == '__main__':

    train_doc_ids, train_categories, train_docs = load_categories(
        "./smart-dataset/datasets/DBpedia/smarttask_dbpedia_train.json")
    test_doc_ids, test_categories, test_docs = load_categories(
        "./smart-dataset/datasets/DBpedia/smarttask_dbpedia_test.json")
    
    print("Preprocessing...")
    processed_train_docs = preprocess_multiple(train_docs)
    processed_test_docs = preprocess_multiple(test_docs)
    print(processed_train_docs[0])
    print(len(processed_test_docs))

    print("Getting vocabulary...")
    vocabulary = load_vocabulary(processed_train_docs)
    print("Getting feature vectors...")
    train_feature_vectors = get_feature_vectors(
        processed_train_docs, vocabulary)
    test_feature_vectors = get_feature_vectors(processed_test_docs, vocabulary)

    print("Training...")
    classifier = train(train_feature_vectors, train_categories)

    print("Applying model on test data...")
    predicted_categories = classifier.predict(test_feature_vectors)

    print("Evaluating")
    precision = evaluate(test_categories, predicted_categories)

    print(f"Precision:\t{precision}")
