import sys, json, csv
import types
from typing import Callable, Dict, List, Set, Tuple
from sklearn.ensemble import RandomForestRegressor
import gensim.downloader as api
import gensim
sys.path.insert(1, './advanced-type-prediction/extract_features')
from extract_No13to15_features import TypeTaxonomy, extract_features_13to15
from extract_No16_feature import extract_features_16
from extract_No17to19_features import extract_features_17to19
from extract_No20to22_features import extract_features_20to22
#from extract_No23to25_features import extract_features_23to25
from extract_No23to25_features_optimized import extract_features_23to25


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_types(path):
    types = []
    with open(path) as f:
        reader = csv.reader(f,delimiter='\t',)
        next(reader, None) 
        for row in reader:
            types.append(row[0])

    return types

def extract_features(
    question:str,
    dp_type: str,
    DBpedia_map_type_entities:Dict,
    docID_DBOtype_dict:Dict,
    typeobj:TypeTaxonomy,
    training_map_type_questions:Dict,
    model_loaded:gensim.models.keyedvectors.KeyedVectors,
    es: Elasticsearch,
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
    features_1to5 = extract_features_1to5(DBpedia_map_type_entities,dp_type,question,es)
    feature_vect = list(features_1to5.values())

    features_11to12 = extract_features_11to12(dp_type, question,docID_DBOtype_dict,es)
    feature_vect.extend(list(features_11to12.values()))

    features_13to15 = extract_features_13to15(typeobj,dp_type)
    feature_vect.extend(list(features_13to15.values()))
    
    features_16=extract_features_16(DBpedia_map_type_entities,dp_type)
    feature_vect.extend(list(features_16.values()))
    
    features_17to19=extract_features_17to19(training_map_type_questions,dp_type,question)
    feature_vect.extend(list(features_17to19.values()))
    
    features_20to22=extract_features_20to22(dp_type,question)
    feature_vect.extend(list(features_20to22.values()))
    
    features_23to25=extract_features_23to25(model_loaded,dp_type, question, mode="Euclidean")
    feature_vect.extend(list(features_23to25.values()))

    if add_extra_features:
        features_23to25_variant=extract_features_23to25(model_loaded,dp_type, question, mode="similarities")
        feature_vect.extend(list(features_23to25_variant.values()))
    
    return feature_vect

def train(X, y):
    return RandomForestRegressor().fit(X,y)

def dump_results(results,results_path):
    with open(results_path, 'w+') as f:
        json.dump(results, f)

def type_prediction(train_path, test_path, types_path, results_path):
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    types = load_types(types_path)
    no_types = len(types)
    train_features = []
    train_types = []
    print('Extracting feature vectors for train...')
    for row in train_data:
        if row['category'] != 'resource' or len(row['type']) == 0 or not row['question']:
            continue
        for type in types:
            if type == row['type'][0]:
                train_types.append(1.0)
            else:
                train_types.append(0.0)
            features = extract_features(row['question'], type)
            train_features.append(features)

    test_features = []
    test_ids = []
    real_test_ids = []
    print('Extracting feature vectors for test...')
    for row in test_data:
        real_test_ids.append(row['id'])
        if row['category'] != 'resource':
            continue
        test_ids.append(row['id'])
        for type in types:
            features = extract_features(row['question'], type)
            test_features.append(features)

    print('Training...')
    classifier = train(train_features,train_types)

    print('Predicting...')
    predicted_type_arrays = classifier.predict(test_features)

    top_10_types = []
    print('Populating types...')
    for i in range(len(test_ids)):
        probs = {}
        for j in range(no_types):
            type = types[j]
            probs[type] = predicted_type_arrays[i*no_types+j]
        list_probs = [{'type':type,'prob': prob} for type, prob in sorted(probs.items(), key=lambda item: item[1])]
        top_10_types.append(list_probs)
    
    for i,id in enumerate(test_ids):
        id_index = real_test_ids.index(id)
        test_data[id_index]['type'] = top_10_types[i][0]['type']
    print('Dumping results to "{}"...'.format(results_path))
    dump_results(test_data, results_path)

if __name__ == '__main__':
    type_prediction("smart-dataset/datasets/DBpedia/smarttask_dbpedia_train.json","./category_results.json","./smart-dataset/evaluation/dbpedia/dbpedia_types.tsv", "./results.json")