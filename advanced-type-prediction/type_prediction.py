import sys, json, csv
import types
from typing import Callable, Dict, List, Set, Tuple
from sklearn.ensemble import RandomForestRegressor
# import gensim.downloader as api
# import gensim
sys.path.insert(1, './advanced-type-prediction/extract_features')
# from extract_No13to15_features import TypeTaxonomy, extract_features_13to15
# from extract_No16_feature import extract_features_16
# from extract_No17to19_features import extract_features_17to19
# from extract_No20to22_features import extract_features_20to22
# #from extract_No23to25_features import extract_features_23to25
# from extract_No23to25_features_optimized import extract_features_23to25


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

def extract_features(question, type):
    # ft13to15 = extract_features_13to15(question, type)
    # ft16 = extract_features_16(question, type)
    # ft17to19 = extract_features_17to19(question, type)
    # ft20to22 = extract_features_20to22(question, type)
    # ft23to25 = extract_features_23to25(question, type)
    # features = {**ft13to15, **ft16, **ft17to19, **ft20to22, **ft23to25}.values()
    features =[0]*10
    if type == "dbo:Place" and "place" in question:
        features[1] = 1.2

    return features

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
        test_data[id_index]['type'] = top_10_types[i][0]
    print('Dumping results to "{}"...'.format(results_path))
    dump_results(test_data, results_path)

if __name__ == '__main__':
    type_prediction("smart-dataset/datasets/DBpedia/smarttask_dbpedia_train.json","./category_results.json","./smart-dataset/evaluation/dbpedia/dbpedia_types.tsv", "./results.json")