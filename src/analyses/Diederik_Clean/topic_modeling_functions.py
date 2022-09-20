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



def load_lda_model(INPUT_DIR, LOAD_VIS=True, LOAD_DICT=True, LOAD_TEXTS=True, LOAD_COHERENCE_SCORE=False):
    """
    --> function that loads an LDA model.

        Parameters:
        -----------
            INPUT_DIR: Str -> input directory path where the model resides
            LOAD_VIS: Bool (default = True) -> load pyLDAvis visualisation
            LOAD_DICT:  Bool (default = True) -> load gensim.corpora.dictionary.Dictionary
            LOAD_TEXTS:  Bool (default = True) -> load lemmatised_documents
            LOAD_COHERENCE_SCORE: Bool (default = False) -> calculate coherence score

    """

    name = os.path.basename(INPUT_DIR)
    output = {'lda_model': None, 'coherence_score': None, 'visualisation': None, 'texts': None, 'dictionary': None}

    if os.path.exists(INPUT_DIR):
        files = os.listdir(INPUT_DIR)

        for file in files:
            path = os.path.abspath(os.path.join(INPUT_DIR, file))

            if file.endswith('.model'):
                output['lda_model'] = gensim.models.LdaModel.load(path)
            elif file.endswith('.html') and LOAD_VIS:
                output['visualisation'] = HTML(filename=path)
            elif file.endswith('.dict') and LOAD_DICT:
                output['dictionary'] = Dictionary.load(path)
            elif file.endswith('.pickle') and LOAD_TEXTS:
                with open(path, 'rb') as fp:
                    output['texts'] = pickle.load(fp)
            elif file.endswith('.txt'):
                output['coherence_score'] = file[:-4]

    if LOAD_COHERENCE_SCORE and output['coherence_score'] is None:
        if output['texts'] is None:
            raise Exception("LOAD_TEXTS=True Parameter and .pickle file is required to calculate the coherence score.")
        if output['dictionary'] is None:
            raise Exception("LOAD_DICT=True Parameter and .pickle file is required to calculate the coherence score.")

        output['coherence_score'] = calculate_coherence_score(MODEL=output['lda_model'], LEMMATIZED_TEXT=output['texts'], DICTIONARY=output['dictionary'], COHERENCE='c_v')

    return(output)