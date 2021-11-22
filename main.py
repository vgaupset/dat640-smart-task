import sys,csv,json
sys.path.insert(1, "./baseline-category-prediction-2")
sys.path.insert(1, "./advanced-type-prediction")
sys.path.insert(1, "./advanced-type-prediction/extract_features")
sys.path.insert(1, "./advanced-type-prediction/util")
sys.path.insert(1, "./smart-dataset/evaluation/dbpedia")

from category_prediction import category_prediction
from type_prediction import type_prediction
import evaluate

def load_json_data(path):
    with open(path) as f:
        data = json.load(f)
    return data

def dump_json_data(data, path):
    with open(path, 'w+') as f:
        json.dump(data, f)

def dump_advanced_results(base_path, advanced_results_csv_path, dump_path):
    base = load_json_data(base_path)
    types = []
    ids = []
    with open(advanced_results_csv_path, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i % 2 == 0:
                ids.append(row[0])
            else:
                types.append(row)
    for entry in base:
        if entry['id'] in ids and entry['category'] == 'resource':
            id_index = ids.index(entry['id'])
            entry['type'] = types[id_index]
    dump_json_data(base, dump_path)

def dump_baseline_results(base_path, baseline_results_json_path, dump_path):
    base = load_json_data(base_path)
    baseline_results_json = load_json_data(baseline_results_json_path)
    for entry in base:
        if entry['id'] in baseline_results_json and entry['category'] == 'resource':
            entry['type'] = baseline_results_json[entry['id']]
    dump_json_data(base, dump_path)


if __name__ == '__main__':
    path_to_test = "./smart-dataset/datasets/DBpedia/smarttask_dbpedia_test.json"
    category_prediction(
        ["./smart-dataset/datasets/DBpedia/smarttask_dbpedia_train.json",
         "./smart-dataset/datasets/Wikidata/lcquad2_anstype_wikidata_train.json"],
        [path_to_test]
    )
    type_hierarchy, max_depth = evaluate.load_type_hierarchy('smart-dataset/evaluation/dbpedia/dbpedia_types.tsv')
    ground_truth = evaluate.load_ground_truth(path_to_test, type_hierarchy)
    system_output_category = evaluate.load_system_output('./category_results.json')

    type_prediction(filepath_training_types="./advanced-type-prediction/data/training_types.json", 
                    filename_trained_model = './advanced-type-prediction/data/finalized_model.sav', 
                    filepath_for_train="./advanced-type-prediction/data/for_training_type_label_alltypes.csv", 
                    filepath_baseline="./advanced-type-prediction/data/baseline_result.json",
                    filepath_testing="./smart-dataset/datasets/DBpedia/smarttask_dbpedia_test.json",
                    result_path="./advanced-type-prediction/data/advanced_results_3.csv")

    dump_baseline_results('./category_results.json','./advanced-type-prediction/data/baseline_result.json', './baseline_results.json')
    system_output_baseline = evaluate.load_system_output('./baseline_results.json')
    dump_advanced_results('./baseline_results.json','./advanced-type-prediction/data/advanced_results_3.csv', './advanced_results.json')
    system_output_advanced = evaluate.load_system_output('./advanced_results.json')



    print('\n\n\033[32mCategory Prediction - literal, boolean results:\033[0m')
    evaluate.evaluate(system_output_category, ground_truth, type_hierarchy, max_depth)
    print('\n\n\033[32mType Prediction - Baseline results:\033[0m')
    evaluate.evaluate(system_output_baseline, ground_truth, type_hierarchy, max_depth)
    print('\n\n\033[32mType Prediction - Advanced results:\033[0m')
    evaluate.evaluate(system_output_advanced, ground_truth, type_hierarchy, max_depth)
