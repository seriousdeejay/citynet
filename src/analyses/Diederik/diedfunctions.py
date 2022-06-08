import pandas as pd # only required for old 'lemmatize' function

# create_lemmatised_wordlists()
from tqdm.notebook import tqdm
import spacy
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_lg")
import re

# import_lemmatised_wordlists()
import pickle
import os

#   vectorize()
from gensim.corpora import Dictionary

# initialize_wordcloud()
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from wordcloud import WordCloud

# Calculate_corpus
from collections import Counter
import pandas as pd

import math

### Can be removed in the near future
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



def create_lemmatised_wordlists(INPUT_DIR="../../../../data/enwiki_city_pairs/", OUTPUT_DIR= "../../../../data/NOUN/", POS=['NOUN'], OVERRIDE_OLD_WORDLISTS = False):
    """
    -->
        function that function that lemmatises text files.

        Parameters:
        -----------
            INPUT_PATH: str -> path to text files
            OUTPUT_DIR: str -> path to lemmatised wordlists
            POS: list (default = ['NOUN']) -> options: (https://spacy.io/usage/spacy-101#annotations-pos-deps)
    """

    #Checks if valid part-of-speech list was provided
    if not isinstance(POS, list):
        raise Exception("POS needs to be a list!")

    POStags=["PROPN", "AUX", "NOUN", "ADJ", "VERB", "ADP", "SYM", "NUM"]
    for tag in POS:
        if tag.upper() not in POStags:
            raise Exception(f'POSfilter only allows a list with one or multiple from the following tags: {POStags}.')
   
    for root, dirs, files in tqdm(list(os.walk(INPUT_DIR))):
        
        # Create subdirectories in output path
        [os.makedirs(os.path.join(OUTPUT_DIR, dir), exist_ok=True) for dir in dirs]

        for file in tqdm(files, total=len(files)):
            file_path = os.path.join(root, file)
            file_output_dir = root.replace(INPUT_DIR, OUTPUT_DIR)
            if OVERRIDE_OLD_WORDLISTS or not os.path.exists(os.path.join(file_output_dir, f"{''.join(POS)}__{file[:-4]}__.pickle" )):
                lemmatize_file(file_path, file_output_dir, city_pair=file[:-4], POS=POS)

    return('Succesfully lemmatised the given texts.')



def lemmatize_file(FILE_PATH, OUTPUT_DIR, city_pair, POS):
    """
    -->
        function that lemmatises a single file.

        Parameters:
        -----------
            FILE_PATH: str -> path to text file
            OUTPUT_DIR: str -> path to lemmatised wordlist
            city_pair: str -> names of the two cities
            POS: list (default = ['NOUN']) -> options: (https://spacy.io/usage/spacy-101#annotations-pos-deps)
    """
    with open(FILE_PATH, 'r', encoding='utf-16') as f:
        city_pair_text_list = [x.strip().lower() for x in f.read().split('\n') if len(x) and 'title=' not in x]
    print(len(city_pair_text_list))
    
    nr_of_chunks = len(city_pair_text_list)//1000 + 1
    chunk_size = (len(city_pair_text_list)-1)//nr_of_chunks
    chunked_text = [' '.join(city_pair_text_list[offs:offs+chunk_size]) for offs in range(0, len(city_pair_text_list), chunk_size)]
    
    processed_text = [text for text in tqdm(nlp.pipe(chunked_text, n_process=2, batch_size=1, disable=["ner", "parser"]), total=len(chunked_text))]
    lemmatized_text = [[word.lemma_ for word in text if word.pos_ in POS and not word.is_punct and not word.is_stop] for text in processed_text]
    regexed_text = [[re.sub(r'\W+', '', word) for word in text] for text in lemmatized_text]
    flattened_words = [item for sublist in regexed_text for item in sublist]
    
    with open(f"{OUTPUT_DIR}/{''.join(POS)}__{city_pair}__.pickle", 'wb') as fp:
        pickle.dump(flattened_words, fp)



def import_lemmatised_wordlists(PATH='../../../../data/enwiki_city_pairs_lemmatised/NOUN/', sort=True):
    """
    -->
        function that imports (POS specific) wordlists belonging to specific city pairs.

        Parameters:
        -----------
            PATH: str -> path to lemmatised wordlists
            sort: bool (default = True) -> sort based on the number of words
    """
    
    if not os.path.isdir(PATH):
        raise Exception("Path is incorrect.")

    data = []

    for root, dirs, files in os.walk(PATH, topdown=True):
        for name in files:
            file_path = os.path.join(root, name)
            parent_dir = os.path.basename(os.path.dirname(file_path))

            with open(file_path, 'rb') as fp:
                data.append((pickle.load(fp), parent_dir, name.split('__')[1]))
    
    if sort:
        data = sorted(data, key=lambda x: len(x[0]), reverse=True)

    return data



# Vectorization
def vectorize(lemmatized_text, filter_extremes=True, MIN_DF = 1, MAX_DF = 0.6):
    """
    -->
        function that vectorizes preprocessed (lemmatized) text.

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



def initialize_wordcloud(background_color='#fff', max_words=200):
    """
    -->
        function that initialised a wordcloud.

        Parameters:
        -----------
            background_color: hex (e.g. #fff) -> background color of the wordcloud

    """

    # black circle
    response = requests.get('https://i.ibb.co/kHNWRYD/black-circle-better.png')
    circle_mask = np.array(Image.open(BytesIO(response.content)))

    wordcloud = WordCloud(max_words=max_words,
                    background_color=background_color,
                    # font_path='/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf',
                    color_func=lambda *args, **kwargs: (0,0,0),
                    mask=circle_mask)
    
    return wordcloud



def calculate_corpus(data: list, CORPUS_PATH='./new_corpus.csv'):
    """
    -->
        function that creates a corpus from a list of wordlists. 

        Parameters:
        -----------
            data: list of tuples (wordlist, parent_dir, city_pair) -> output from 'import_lemmatised_wordlists()'.
            CORPUS_PATH: str (default = './new_corpus.csv') -> path of the output file.

    """
    corpus = pd.DataFrame(columns=['word'])

    # Create Dataframe with Word Counts
    for i in tqdm(range(len(data))):
        word_count = Counter(data[i][0])

        new_word = list(set(word_count.keys()) - set(corpus['word']))
        corpus = corpus.append(pd.DataFrame({'word': new_word}), ignore_index=True)

        wordlist = []
        for word in corpus['word']:
            if word in word_count.keys():
                wordlist.append(word_count[word])
            else:
                wordlist.append(0)

        corpus[data[i][2]] = wordlist

    corpus.set_index('word', inplace=True)
    corpus.fillna(0, inplace=True)

    corpus.to_csv(CORPUS_PATH)
        
    return corpus



def calculate_tf(corpus='', OUTPUT_PATH='', include_idf=False, min_freq=0, min_doc=0) -> tuple:
    """
    -->
        function that calculates the term frequency (and optionally inverse document frequency) for a corpus

        Parameters:
            -----------
                corpus: pd.DataFrame -> output from 'calculate_corpus()'.
                OUTPUT_PATH: str -> path of the output file.
                include_idf: bool ( default=False) -> Whether you would like to include -idf.

    """
    if not isinstance(min_freq, int) or not min_freq:
        raise Exception("min_freq should be given an integer value greater than 0.")
    
    if not isinstance(min_doc, int) or not min_doc:
        raise Exception("min_doc should be given an integer value greater than 0.")
        
    if not isinstance(corpus, pd.DataFrame):
        raise Exception("Corpus is required as input (can use calculate_corpus to create one.)")
        
    if not OUTPUT_PATH:
        OUTPUT_PATH = './new_tfidf.csv' if include_idf else './new_tf.csv'
        
    try:
        corpus.set_index('word', inplace=True)
    except:
        pass
    
    tf_idf = {k: [] for k in corpus.columns}
    sumdict = {x: sum(corpus[x]) for x in corpus.columns}
    
    # Create Dataframe with Relative Word Frequencies
    for index, row in tqdm(corpus.iterrows(), total=len(corpus)):
        docs_with = np.count_nonzero(row)

        for colname, count in row.items():
            if docs_with >= min_doc and count >= min_freq:
                # total_uniques = np.count_nonzero(corpus[colname])
                # total_words = sum(corpus[colname])
                tf = count / sumdict[colname]
                result = tf
                
                if include_idf:
                    idf = math.log(len(corpus.columns) / docs_with)
                    result *= idf 
            else:
                result = 0

            tf_idf[colname].append(result)

    tf_idf_df = pd.DataFrame.from_dict(tf_idf)
    tf_idf_df.set_index(corpus.index, inplace=True)

    tf_idf_df.to_csv(OUTPUT_PATH)
    print(f"Saved results to '{OUTPUT_PATH}'.")

    return tf_idf_df



def check_path(path):
    """
    -->
        function that checks if file exists and if so, asks the user if they want to override it.

        Parameters:
            -----------
                path: str -> path to check

    """
    if os.path.exists(path):
        protection = input(f"This file already exists. Are you sure you want to override?\nType 'Yes' to continue: ")
        if protection == 'Yes':
            return True
        else:
            print("\nCanceling Operation...\n")
            return False
    else:
        return True