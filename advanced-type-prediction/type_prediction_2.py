import sys, json, csv, pickle
import types
from typing import Callable, Dict, List, Set, Tuple
#from sklearn.ensemble import RandomForestRegressor
import gensim.downloader as api
import gensim
sys.path.insert(1, 'extract_features')
sys.path.insert(1, 'util')
import os.path
# from helper_function import preprocess
# from extract_No17to19_features import get_analyze,extract_features_17to19
# from extract_No20to22_features import extract_features_20to22
# from extract_No23to25_features_optimized import extract_features_23to25
from train_and_predict import load_training_data,PointWiseLTRModel, get_rankings


def type_prediction(filepath_training_types="data/training_types.json", 
                    filename_trained_model = 'data/finalized_model.sav', 
                    filepath_for_train="data/for_training_type_label.csv", 
                    filepath_baseline="data/baseline_result.json",
                    filepath_testing="../smart-dataset/datasets/DBpedia/smarttask_dbpedia_test.json",
                    result_path="data/advanced_results_2.csv"):
   
    
    if os.path.isfile(result_path):
        return None
    
    with open(filepath_training_types,encoding='utf-8') as json_file:
        training_map_type_questions = json.load(json_file)
    try:
        model_loaded = gensim.models.keyedvectors.KeyedVectors.load('googleNews.d2v')
    except:
        model_loaded = api.load('word2vec-google-news-300')
        model_loaded.save('googleNews.d2v')
        model_loaded = gensim.models.keyedvectors.KeyedVectors.load('googleNews.d2v')

    
    try:
        ltr=PointWiseLTRModel()
        ltr.model = pickle.load(open(filename_trained_model, 'rb'))
    except:
        X_train,y_train=load_training_data(filepath_for_train)
        ltr=PointWiseLTRModel()
        ltr._train(X_train,y_train)
        print("finish training")
        # save the model to disk
        filename = 'data/finalized_model.sav'
        pickle.dump(ltr.model, open(filename_trained_model, 'wb'))
        print("trained model has been saved")
        # load the model from disk
        ltr.model = pickle.load(open(filename, 'rb'))
        print("trained model has been loaded")
        
        get_rankings(ltr,
                    training_map_type_questions,
                    model_loaded,model_loaded,
                    filepath_baseline,
                    filepath_testing,
                    result_path)
                       

if __name__ == '__main__':
    type_prediction()