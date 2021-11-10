import json
import string
import re
from typing import Callable, Dict, List, Set, Tuple
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
    #return doc.lower()
    return doc

def analyze_query(
    es: Elasticsearch, query: str, field: str, index: str = "toy_index"
) -> List[str]:
    """Analyzes a query with respect to the relevant index.

    Args:
        es: Elasticsearch object instance.
        query: String of query terms.
        field: The field with respect to which the query is analyzed.
        index: Name of the index with respect to which the query is analyzed.

    Returns:
        A list of query terms that exist in the specified field among the
        documents in the index.
    """
    tokens = es.indices.analyze(index=index, body={"text": query})["tokens"]
    query_terms = []
    for t in sorted(tokens, key=lambda x: x["position"]):
        # Use a boolean query to find at least one document that contains the
        # term.
        hits = (
            es.search(
                index=index,
                query={"match": {field: t["token"]}},
                _source=False,
                size=1,
            )
            .get("hits", {})
            .get("hits", {})
        )
        doc_id = hits[0]["_id"] if len(hits) > 0 else None
        if doc_id is None:
            continue
        query_terms.append(t["token"])
    return query_terms


def get_doc_term_freqs(
    es: Elasticsearch, doc_id: str, field: str, index: str
) -> Dict[str, int]:
    """Gets the term frequencies of a field of an indexed document.

    Args:
        es: Elasticsearch object instance.
        doc_id: Document identifier with which the document is indexed.
        field: Field of document to consider for term frequencies.
        index: Name of the index where document is indexed.

    Returns:
        Dictionary of terms and their respective term frequencies in the field
        and document.
    """
    tv = es.termvectors(
        index=index, id=doc_id, fields=field, term_statistics=True
    )
    if tv["_id"] != doc_id:
        return None
    if field not in tv["term_vectors"]:
        return None
    term_freqs = {}
    for term, term_stat in tv["term_vectors"][field]["terms"].items():
        term_freqs[term] = term_stat["term_freq"]
    return term_freqs


class Indexer:
    def __init__(self,index: str,index_settings:dict, reset=True):
        #self._filepath = filepath
        self.dictionary = {}     
        self.index = index
        self.index_settings= index_settings
        es = Elasticsearch()
        es.info()
        self.es = es
        if reset:
            self.reset_index()

    
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

    def reset_index(self) -> None:
        """Clears index"""
        if self.es.indices.exists(self.index):
            self.es.indices.delete(index=self.index)

        self.es.indices.create(index=self.index, body=self.index_settings)
    
    def bulk_index(self,data) -> None:
        """Indexes documents from JSONL file."""
        bulk_data = []
        for item in data:
            bulk_data.append(
                {"index": {"_index": self.index, "_id": item.pop("id")}}
            )
            bulk_data.append(item)
        self.es.bulk(index=self.index, body=bulk_data, refresh=True)
    
    def check_esIndex_count(self)->int:
        self.es.indices.refresh(self.index)
        count = self.es.cat.count(self.index, params={"format": "json"})
        return int(count[0]["count"])

    def check_esIndex_content(self, id:str):
        return self.es.get(index=self.index, id=id)