import json
import string
import re
from elasticsearch import Elasticsearch

def load_dict_from_json(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except:
        print(f'File \'{filename}\' not found.')
        return None

def save_dict_to_json(doc, filename):
    with open(filename, 'w') as f:
        json.dump(doc, f)


def reset_index(es: Elasticsearch, index_name: str, index_settings) -> None:
    """Clears index"""
    if es.indices.exists(index_name):
        es.indices.delete(index=index_name)

    es.indices.create(index=index_name, body=index_settings)

def preprocess(doc: str) -> str:
    """Preprocesses text to prepare it for feature extraction.

    Args:
        doc: String comprising the unprocessed contents of some email file.

    Returns:
        String comprising the corresponding preprocessed text.
    """
    re_html = re.compile("<[^>]+>")
    doc = re_html.sub(" ", doc)
    #remove pure digits 
    doc=re.sub(r"(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b","",doc)
    # Replace punctuation marks (including hyphens) with spaces.
    for c in string.punctuation:
        doc = doc.replace(c, " ")
    return doc.lower()



class indexer:
    def __init__(self,filename):
        self._filename = filename
        self._dictionary = {}
    
    def preprocess(self, doc: str) -> str:
        """Preprocesses text to prepare it for feature extraction.

    Args:
        doc: String comprising the unprocessed contents of some email file.

    Returns:
        String comprising the corresponding preprocessed text.
    """
        re_html = re.compile("<[^>]+>")
        doc = re_html.sub(" ", doc)
        #remove pure digits 
        doc=re.sub(r"(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b","",doc)
        # Replace punctuation marks (including hyphens) with spaces.
        for c in string.punctuation:
            doc = doc.replace(c, " ")
        return doc.lower()

    def reset_index(es: Elasticsearch, index_name: str, index_settings) -> None:
        """Clears index"""
        if es.indices.exists(index_name):
            es.indices.delete(index=index_name)

        es.indices.create(index=index_name, body=index_settings)