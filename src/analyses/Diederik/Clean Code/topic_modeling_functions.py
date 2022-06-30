import os
import pickle

# Vectorize
from gensim.corpora import Dictionary


# Train_lda_model
from gensim.models.wrappers import LdaMallet

#Compare_lda_models
from tqdm.notebook import tqdm

# Calculate_coherence_score
from gensim.models import CoherenceModel

# Save_lda_model
import gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pandas as pd

# Load_lda_model
from IPython.display import HTML



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



def calculate_coherence_score(MODEL, LEMMATIZED_TEXT, DICTIONARY, COHERENCE='c_v'):
    """
    --> function that calculates the coherence score of an LDA model.

        Parameters:
        -----------
            MODEL: LdaMallet -> LDA model to save
            LEMMATIZED_TEXT: list -> lemmatised documents
            DICTIONARY: gensim.corpora.dictionary.Dictionary -> Dictionary
            MIN_DF: int (default = 1) -> minimum document frequency
            MAX_DF: int (default = 0.6) -> maximum document frequency

    """
    coherence_score = CoherenceModel(model=MODEL, texts=LEMMATIZED_TEXT, dictionary=DICTIONARY, coherence=COHERENCE).get_coherence()
    return coherence_score



def train_lda_model(lemmatized_text, dictionary=[], corpus=[], MIN_DF = 0.05, MAX_DF = 0.9, N_TOPICS = 10, N_ITERATIONS = 1000, PATH_TO_MALLET=r'C:/mallet/bin/mallet.bat', GET_COHERENCE_SCORE=True, COHERENCE='c_v'):
    """
    --> function that trains model.

        Parameters:
        -----------
            lemmatized_text: list, str -> contains the key words to be matched (created with the lemmatization function)
            dictionary: gensim.corpora.dictionary.Dictionary -> output from vectorization function
            corpus: list ([dictionary.doc2bow(text)...) -> output from vectorization function
            MIN_DF: int (default = 1) -> minimum document frequency
            MAX_DF: int (default = 0.6) -> maximum document frequency
            N_TOPICS: int (default = 10) -> Topics to detect
            N_ITERATIONS: int (default = 1000) -> 1000 often enough
            PATH_TO_MALLET: str (default = 'C:/mallet/bin/mallet.bat') -> location of your mallet execution file
            GET_COHERENCE_SCORE: Bool (default = True) -> Whether the coherence should be calculated
            COHERENCE: Str ( default = 'c_v') -> ..

    """

    # Call vectorization function if either dictionary or corpus is missing as parameter
    if not type(dictionary) == gensim.corpora.dictionary.Dictionary or not corpus:
        dictionary, corpus = vectorize(lemmatized_text, MIN_DF, MAX_DF)

    print(f"topics: {N_TOPICS}, MIN_DF: {MIN_DF}, MAX_DF: {MAX_DF}")

    lda_model = LdaMallet(PATH_TO_MALLET,
                corpus=corpus,
                id2word=dictionary,
                num_topics=N_TOPICS,
                # alpha=auto,
                optimize_interval=10,
                iterations=N_ITERATIONS)
    

    output = {'lda_model': lda_model, 'coherence_score': None, 'dictionary': dictionary, 'corpus': corpus}

    if GET_COHERENCE_SCORE:
        output['coherence_score'] = calculate_coherence_score(MODEL=lda_model, LEMMATIZED_TEXT=lemmatized_text, DICTIONARY=dictionary, COHERENCE=COHERENCE)

    return(output)



def save_lda_model(MODEL,  OUTPUT_DIR, NAME, COHERENCE_SCORE=None, DICTIONARY=None, CORPUS=None, TEXTS=None, VIS=None, SAVE_VIS=True, SAVE_DICT=True, SAVE_TEXTS=True, SAVE_COHERENCE_SCORE=True):
    """
    --> function that saves an LDA model.

        Parameters:
        -----------
            MODEL: LdaMallet -> LDA model to save
            NAME: str -> Name of the model
            OUTPUT_DIR: Str -> Output directory path where the model should be saved to
            DICTIONARY: gensim.corpora.dictionary.Dictionary -> Dictionary
            Corpus: corpus -> corpus
            TEXTS: list -> lemmatised documents
            VIS: pyLDAvis -> visualisation of the topics
            SAVE_VIS: Bool (default = True) ->  save visualisation
            SAVE_DICT: Bool (default = True) -> save dictionary
            SAVE_TEXTS: Bool (default = True) -> save lemmatised documents

    """
    directory = os.path.join(OUTPUT_DIR, NAME)
    os.makedirs(directory, exist_ok=False)
    
    # Type checks
    valid_model = isinstance(MODEL, gensim.models.wrappers.ldamallet.LdaMallet)
    valid_corpus = isinstance(CORPUS, list)
    valid_dictionary = isinstance(DICTIONARY, gensim.corpora.dictionary.Dictionary)
    valid_texts = isinstance(TEXTS,  (pd.Series, list))
    
    
    if not valid_model:
        raise Exception("The model you provided is not a valid LdaMallet model.")
    
    if SAVE_VIS and not VIS and not (valid_corpus and valid_dictionary):
        raise Exception("Creating and saving the visualisation requires CORPUS and DICTIONARY as parameters.")
    
    if SAVE_DICT and not valid_dictionary:
                raise Exception("Dictionary parameter is not of type gensim.corpora.dictionary.Dictionary.")

    if SAVE_TEXTS and not valid_texts:
            raise Exception("TEXTS parameter is not a valid list type.")
    
    # Actual Saving
    print('Saving lda model...')
    
    if SAVE_VIS:
        if not VIS:
            lda_conv = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(MODEL) # my_models[0]['lda_model__2']['model']
            VIS = gensimvis.prepare(lda_conv, CORPUS, DICTIONARY)
            
        pyLDAvis.save_html(VIS, os.path.join(directory, f"{NAME}_visualisation.html"))        
    
    if SAVE_DICT:
        DICTIONARY.save(os.path.join(directory, f"{NAME}_dictionary.dict"))
    
    if SAVE_TEXTS:
        if isinstance(TEXTS, list):
            with open(os.path.join(directory, f"{NAME}_texts.pickle"), 'wb') as fp:
                pickle.dump(TEXTS, fp)
        else:
            TEXTS.to_pickle(os.path.join(directory, f"{NAME}_texts.pickle"))

    MODEL.save(os.path.join(directory, f"{NAME}_model.model"))

    if SAVE_COHERENCE_SCORE:
        if not isinstance(COHERENCE_SCORE, float):
            if not (valid_dict and valid_texts):
                raise Exception("Creating and saving the coherence score requires valid DICTIONARY and TEXTS parameters.")
                
            print("Calculating coherence score...")
            COHERENCE_SCORE = calculate_coherence_score(MODEL=MODEL, LEMMATIZED_TEXT=TEXTS, DICTIONARY=DICTIONARY, COHERENCE='c_v')

        with open(os.path.join(directory, f"{COHERENCE_SCORE}.txt"), 'w') as fp:
            pass

    print(f'Model has been saved to the following location: {directory}.')

    return MODEL



def compare_lda_models(OUTPUT_DIR, TOPIC_SELECTION, LEMMATIZED_TEXT, DICTIONARY=[], CORPUS=[], MIN_DF=0.05, MAX_DF=0.9, N_ITERATIONS=1000, PATH_TO_MALLET=r'C:/mallet/bin/mallet.bat', GET_COHERENCE_SCORE=True, COHERENCE='c_v'):
    """
    --> function that creates LDA models with a multitude of topic numbers.

        Parameters:
        -----------
            OUTPUT_DIR: Str -> Output directory path where the models should be saved to
            TOPIC_SELECTION: range -> list of numbers that specify the amount of topics to detect
            LEMMATIZED_TEXT: list -> lemmatised documents
            DICTIONARY: gensim.corpora.dictionary.Dictionary -> Dictionary
            CORPUS: corpus -> corpus
            MIN_DF: int (default = 0.05) -> minimum document frequency
            MAX_DF: int (default = 0.9) -> maximum document frequency
            N_ITERATIONS: int (default = 1000) -> 1000 often enough
            PATH_TO_MALLET: str (default = 'C:/mallet/bin/mallet.bat') -> location of your mallet execution file
            GET_COHERENCE_SCORE: Bool (default = True) -> Whether the coherence should be calculated
            COHERENCE: Str ( default = 'c_v') -> ..

    """

    if not isinstance(TOPIC_SELECTION, range):
        raise Exception("TOPIC SELECTION needs to be of type 'range()'. e.g. range(2,20,3)")
    
    print(f"Creating lda models for the following number of topics: {[number for number in TOPIC_SELECTION]}.")

    my_models = []

    for N_TOPICS in tqdm(TOPIC_SELECTION, total=len(TOPIC_SELECTION), desc='Creating models...', leave=True):
        output = train_lda_model(lemmatized_text=LEMMATIZED_TEXT, dictionary=DICTIONARY, corpus=CORPUS, MIN_DF = MIN_DF, MAX_DF = MAX_DF, N_TOPICS = N_TOPICS, N_ITERATIONS = N_ITERATIONS, PATH_TO_MALLET=PATH_TO_MALLET, GET_COHERENCE_SCORE=GET_COHERENCE_SCORE, COHERENCE=COHERENCE)
        print('coherence_score:', output['coherence_score'])
        NAME = f"lda_model_{N_TOPICS}topics_{MIN_DF}min_{MAX_DF}max".replace('.', '_')
        save_lda_model(output['lda_model'], COHERENCE_SCORE=output['coherence_score'], DICTIONARY=output['dictionary'], CORPUS=output['corpus'], TEXTS=LEMMATIZED_TEXT, VIS=None, OUTPUT_DIR=OUTPUT_DIR, NAME=NAME, SAVE_VIS=True, SAVE_DICT=True)
        
        # # model, coherence, _, _ = train_model(words, dictionary, corpus, N_TOPICS=n_topics)
        # keyname = f'lda_model__{n_topics}'
        # model.save(f'../../../../lda_models/{keyname}.model')

        my_models.append({'name': NAME, 'lda_model': output['lda_model'], 'N_TOPICS': N_TOPICS, 'coherence_score': output['coherence_score']})
    
    return my_models



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



def load_lda_models(INPUT_DIR, LOAD_VIS=True, LOAD_DICT=True, LOAD_COHERENCE_SCORE=True):
    """
    --> function that loads LDA models.

        Parameters:
        -----------
            OUTPUT_DIR: Str -> Output directory path where the models resides
            LOAD_VIS: Bool (default = True) -> load pyLDAvis visualisation
            LOAD_DICT:  Bool (default = True) -> load gensim.corpora.dictionary.Dictionary
            LOAD_TEXTS:  Bool (default = True) -> load lemmatised_documents
            LOAD_COHERENCE_SCORE: Bool (default = False) -> calculate coherence score
            
    """

    models = []

    if not os.path.exists(INPUT_DIR):
        raise Exception("Invalid Directory Path.")

    for model_directory in os.listdir(INPUT_DIR):
        model_path = os.path.join(INPUT_DIR, model_directory)
        print(model_path)
        models.append(load_lda_model(INPUT_DIR=model_path, LOAD_VIS=LOAD_VIS, LOAD_DICT=LOAD_DICT, LOAD_COHERENCE_SCORE=LOAD_COHERENCE_SCORE))
    
    return models