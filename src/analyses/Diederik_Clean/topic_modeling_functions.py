# Vectorize
from gensim.corpora import Dictionary


def vectorize(lemmatized_text, filter_extremes=True, MIN_DF = 0.05, MAX_DF = 0.9):
    """
    --> function that vectorizes preprocessed (lemmatized) text.

        Parameters:
        -----------
            lemmatized_text: list, str -> contains the key words to be matched
            filter_extremes: bool (default = True) -> whether you want to filter by min/max document frequency
            MIN_DF: int (default = 1) -> minimum document frequency
            MAX_DF: int (default = 0.6) -> maximum document frequency

    """
    
    # Get Vocabulary
    dictionary = Dictionary(lemmatized_text)

    if filter_extremes:
        dictionary.filter_extremes(no_below=MIN_DF, no_above=MAX_DF)
    
    corpus = [dictionary.doc2bow(text) for text in lemmatized_text]
    
    return(dictionary, corpus)