from typing import List
import json
from elasticsearch import Elasticsearch

INDEX_NAME = "smart-task"
INDEX_SETTINGS = {
    "mappings": {
        "properties": {
            "question": {"type": "text", "term_vector": "yes", "analyzer": "english"},
            "type": {"type": "text", "term_vector": "yes", "analyzer": "english"},
            "category": {"type": "text", "term_vector": "yes", "analyzer": "english"},
        }
    }
}
def load_data(path: str) -> List[dict]:
    """ Load json data for smarttask

    Args:
        path: path to json file to load.

    Returns:
        List of dictionaries with keys: id, question, category and type
    """
    f = open(path,)
    data = json.load(f)
    f.close()
    return data

def reset_index(es: Elasticsearch) -> None:
    """Clears index"""
    if es.indices.exists(INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)

def bulk_index(es: Elasticsearch, questions: List[str]) -> None:
    """Iterate over questions and index

    Args:
        es: Elasticsearch instance.
        questions: List of question records.
    """
    indexed_questions = 0
    num_of_questions = len(questions)
    for i, question in enumerate(questions):
        body = {
            'question': question['question'],
            'category': question['category'],
            'type': question['type'],
        }
        if indexed_questions % 100 == 0:
            print("{} percent done".format(i/num_of_questions*100))
        es.index(index=INDEX_NAME, doc_type="_doc", id=question['id'], body=body)
        indexed_questions += 1
    print("{} questions indexed.".format(indexed_questions))


if __name__ == '__main__':
    train_questions = load_data("./smart-dataset/datasets/DBpedia/smarttask_dbpedia_train.json")    
    es = Elasticsearch()
    print(es.info())

    reset_index(es)
    bulk_index(es, train_questions)