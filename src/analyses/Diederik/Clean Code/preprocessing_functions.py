import os
import pickle
from tqdm.notebook import tqdm
import re
import spacy

nlp = spacy.load("en_core_web_lg")
nlp.disable_pipes('ner', 'parser')



def lemmatize_files_paragraphs(INPUT_DIR, OUTPUT_DIR, POS, OVERRIDE_OLD_WORDLISTS, NLP_MAX_LENGTH=1500000):
    """
    -->
        function that lemmatises the paragraphs of a batch of text files.

        Parameters:
        -----------
            INPUT_DIR: Str -> input directory path, to the text files
            OUTPUT_DIR: Str -> output directory path, where you want to save the .pickle files
            POS: list (default = ['NOUN']) -> options: (https://spacy.io/usage/spacy-101#annotations-pos-deps)
            OVERRIDE_OLD_WORDLISTS: Bool -> Whether you want to override existing output files
            NLP_MAX_LENGTH: Int (default: 1500000) -> Allowed number of characters per file
    """
    for root, dirs, files in tqdm(list(os.walk(INPUT_DIR))):

        # Create subdirectories in output path
        [os.makedirs(os.path.join(OUTPUT_DIR, dir), exist_ok=True) for dir in dirs]

        for file in tqdm(files, total=len(files)):
            file_path = os.path.join(root, file)
            file_output_dir = root.replace(INPUT_DIR, OUTPUT_DIR)
            lemmatise_file_paragraphs(FILE_PATH=file_path, FILE_OUTPUT_DIR=file_output_dir, POS=POS, OVERRIDE_OLD_WORDLIST=OVERRIDE_OLD_WORDLISTS, NLP_MAX_LENGTH=NLP_MAX_LENGTH)



def lemmatise_file_paragraphs(FILE_PATH, FILE_OUTPUT_DIR, POS, OVERRIDE_OLD_WORDLIST, NLP_MAX_LENGTH=1500000):
    """
    -->
        function that lemmatises the paragraphs of a single text file.

        Parameters:
        -----------
            FILE_PATH: Str -> input directory path, to the text files
            FILE_OUTPUT_DIR: Str -> output directory path, where you want to save the .pickle files
            POS: list (default = ['NOUN']) -> options: (https://spacy.io/usage/spacy-101#annotations-pos-deps)
            OVERRIDE_OLD_WORDLISTS: Bool -> Whether you want to override existing output files
            NLP_MAX_LENGTH: Int (default: 1500000) -> Allowed number of characters per file
    """

    nlp.max_length = nlp_max_length
    POStags=["PROPN", "AUX", "NOUN", "ADJ", "VERB", "ADP", "SYM", "NUM"]

    #Checks if valid part-of-speech list was provided
    if not isinstance(POS, list):
        raise Exception("POS needs to be a list!")

    for tag in POS:
        if tag.upper() not in POStags:
            raise Exception(f'POSfilter only allows a list with one or multiple from the following tags: {POStags}.')

    CITY_PAIR = os.path.basename(FILE_PATH)[:-4]
    new_path = os.path.join(FILE_OUTPUT_DIR, f"{''.join(POS)}__{CITY_PAIR}__.pickle" )

    if OVERRIDE_OLD_WORDLISTS or not os.path.exists(new_path):
        with open(FILE_PATH, 'r', encoding='utf-16') as f:
            city_pair_paragraphs = [x.strip().lower() for x in f.read().replace('"', "'").replace('“', "'").replace('”', "'").split('\n') if len(x) and 'title=' not in x]

        processed_paragraphs = [text for text in tqdm(nlp.pipe(city_pair_paragraphs, n_process=2, batch_size=1, disable=["ner", "parser"]), total=len(city_pair_paragraphs))]
        lemmatized_paragraphs = [[word.lemma_ for word in paragraph if word.pos_ in POS and not word.is_punct and not word.is_stop] for paragraph in processed_paragraphs]
        regexed_paragraphs= [[re.sub(r'\W+', '', word) for word in paragraph] for paragraph in lemmatized_paragraphs]
        
        with open(new_path, 'wb') as fp:
            pickle.dump(regexed_paragraphs, fp)



def import_lemmatised_wordlists(PATH, sort=True):
    """
    -->
        function that imports (POS specific) wordlists belonging to specific city pairs.

        Parameters:
        -----------
            PATH: str -> path to lemmatised wordlists (e.g. '../../../../data/enwiki_city_pairs_lemmatised/NOUN/')
            sort: bool (default = True) -> sort based on the number of city pair co-occurences
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
    
    # Sort by number of city pair co-occurences
    if sort:
        data = sorted(data, key=lambda x: len(x[0]), reverse=True)

    return data



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



def is_english_word(word, english_words):
    return word.lower() in english_words



def remove_non_existing_words(wordlist: list, english_words) -> list:
    wordset = set(wordlist)
    non_existent = []
    
    for word in wordset:
        if not is_english_word(word, english_words):
            non_existent.append(word)
    # print(non_existent)
    return([word for word in wordlist if word not in non_existent])



def remove_non_existing_words_for_dir(INPUT_DIR, OUTPUT_DIR, english_words_file="../../../input/english_words_alpha_370k.txt"):
    # DIR = "../../../../data/enwiki_city_pairs_lemmatised/VERBNOUNADJ_SPACY_LARGE"
    # NEWDIR = "../../../../data/enwiki_city_pairs_lemmatised/VERBNOUNADJ_SPACY_LARGE_REAL_WORDS"


    for path in [INPUT_DIR, OUTPUT_DIR, english_words_file]:
        if not os.path.exists(path):
            raise Exception("{path} is not a valid path.")

    with open(english_words_file) as word_file:
        english_words = set(word.strip().lower() for word in word_file)

    for file in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, file)
        with open(file_path, 'rb') as fp:
            wordlist = pickle.load(fp)
        
        cleaned_wordlist = remove_non_existing_words(wordlist, english_words)
   
        with open(os.path.join(OUTPUT_DIR, file), 'wb') as fp2:
            pickle.dump(cleaned_wordlist, fp2)
        
        print('old:' len(wordlist), 'new:', len(cleaned_wordlist), 'removed:', len(wordlist)-len(cleaned_wordlist))
