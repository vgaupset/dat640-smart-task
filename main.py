import sys
sys.path.insert(1, "./baseline-category-prediction-2")
sys.path.insert(1, "./advanced-type-prediction")
sys.path.insert(1, "./smart-dataset/evaluation/dbpedia")

from category_prediction import category_prediction
from type_prediction import type_prediction
import evaluate 




if __name__ == '__main__':
    path_to_test = "./smart-dataset/datasets/DBpedia/smarttask_dbpedia_test.json"
    category_prediction(
        ["./smart-dataset/datasets/DBpedia/smarttask_dbpedia_train.json",
         "./smart-dataset/datasets/Wikidata/lcquad2_anstype_wikidata_train.json"],
        [path_to_test]
    )
    type_hierarchy, max_depth = evaluate.load_type_hierarchy('smart-dataset\evaluation\dbpedia\dbpedia_types.tsv')
    ground_truth = evaluate.load_ground_truth(path_to_test, type_hierarchy)
    system_output = evaluate.load_system_output('./category_results.json')
    print('Category results:')
    evaluate.evaluate(system_output, ground_truth, type_hierarchy, max_depth)
    type_prediction("smart-dataset/datasets/DBpedia/smarttask_dbpedia_train.json","./category_results.json","./smart-dataset/evaluation/dbpedia/dbpedia_types.tsv", "./results.json")
    system_output = evaluate.load_system_output('./results.json')
    print('Type results:')
    evaluate.evaluate(system_output, ground_truth, type_hierarchy, max_depth)
