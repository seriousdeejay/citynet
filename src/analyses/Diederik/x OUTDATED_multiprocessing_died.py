import spacy
nlp = spacy.load("en_core_web_sm")
import re
from joblib import Parallel, delayed

nlp.max_length = 1500000
nlp.disable_pipes('ner', 'parser')
POSfilter=["PROPN", "NOUN", "ADJ", "VERB", "ADV"]

def lemmatize_pipe(doc):
    lemma_list = [tok.lemma_.lower() for tok in doc
                  if tok.pos_ in POSfilter and not tok.is_punct and not tok.is_stop] 
    return lemma_list

def preprocess_pipe(texts):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(lemmatize_pipe(doc))
    return preproc_pipe


def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]

def process_chunk(texts):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(lemmatize_pipe(doc))
    return preproc_pipe

def preprocess_parallel(texts, chunksize=2):
    print(chunksize)
    executor = Parallel(n_jobs=7, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)