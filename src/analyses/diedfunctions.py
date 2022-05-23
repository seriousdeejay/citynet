# Imports
from gensim.corpora import Dictionary
from tqdm.notebook import tqdm
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
import re

def lemmatize(texts, POSfilter=["PROPN", "NOUN", "ADJ", "VERB", "ADV"], nlp_max_length = 1500000):
    """
    -->
        function that lemmatizes text(s).

        Parameters:
        -----------
            texts: list, string -> list of strings or string that has to be lemmatized
            POSfilter: list (default=["PROPN", "NOUN", "ADJ", "VERB", "ADV"]) -> lemmatization part-of-speech filter

    """
    POStags=["PROPN", "NOUN", "ADJ", "VERB", "ADV"]
    nlp.max_length = nlp_max_length

    # Checks if user gave their own (list of) Part of Speech tag(s)
    if POSfilter:
        if isinstance(POSfilter, list):
            for POS in POSfilter:
                if POS not in POStags:
                    print(f'POSfilter only allows a list with one or multiple from the following tags: {POStags}.')
                    return
        else:
            print('POSfilter should either be left out or a list of valid POS tags.')
            return
        
    # Gets triggered if a single string is given
    if isinstance(texts, str) and len(texts):
        processed_text = nlp(texts.lower())
        lemmatized_text = [word.lemma_.lower() for word in processed_text if word.pos_ in POSfilter and not word.is_punct and not word.is_stop]
        regexed_text = [re.sub(r'\W+', '', word) for word in lemmatized_text]
    
    # Gets triggered if an array of strings is given
    elif isinstance(texts, (pd.Series, list)) and len(texts):
        processed_text = [text for text in tqdm(nlp.pipe(texts, n_process=-1, disable=["ner", "parser"]), total=len(texts))]
        lemmatized_text = [[word.lemma_.lower() for word in text if word.pos_ in POSfilter and not word.is_punct and not word.is_stop] for text in processed_text]
        regexed_text = [[re.sub(r'\W+', '', word) for word in text] for text in lemmatized_text]
    
    else:
        print('Your provided text could not be processed. Check if the format of your provided text is either a string or a list of strings.')
        return
    
    return regexed_text



# Vectorization
def vectorize(lemmatized_text, MIN_DF = 1, MAX_DF = 0.6):
    """
    -->
        function that vectorizes preprocessed (lemmatized) text.

        Parameters:
        -----------
            lemmatized_text: list, str -> contains the key words to be matched
            MIN_DF: int (default = 1) -> minimum document frequency
            MAX_DF: int (default = 0.6) -> maximum document frequency

    """
    
    # Get Vocabulary
    dictionary = Dictionary(lemmatized_text)
    dictionary.filter_extremes(no_below=MIN_DF, no_above=MAX_DF)
    
    corpus = [dictionary.doc2bow(text) for text in lemmatized_text]
    
    return(dictionary, corpus)