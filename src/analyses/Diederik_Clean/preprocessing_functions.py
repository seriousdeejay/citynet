import os
from ast import literal_eval
import pickle
from tqdm.notebook import tqdm
import re
import spacy

import pandas as pd

nlp = spacy.load("en_core_web_lg")
nlp.disable_pipes('ner', 'parser')

# from general_functions import check_path



def get_english_words(path="../../../input/english_words_alpha_370k.txt"):
    if not os.path.exists(path):
        raise Exception("Provide a valid path to a file with English words.")

    with open(path) as word_file:
            ENGLISH_WORDS = set(word.strip().lower() for word in word_file)
    
    if not len(ENGLISH_WORDS):
        raise Exception("No wordlist could be found in the given file!")
            
    return ENGLISH_WORDS



def is_english_word(word, english_words):
    return word.lower() in english_words



def remove_non_existing_words_from_wordlist(wordlist: list, english_words) -> list:
    if not len(english_words):
        raise Exception("The supplied english words list is empty."
                       )
    wordset = set(wordlist)
    non_existent = []
    
    for word in wordset:
        if not is_english_word(word, english_words):
            non_existent.append(word)
            
    return([word for word in wordlist if word not in non_existent])



def keep_english_words_in_paragraphs(paragraphs, english_words=[], english_words_file="../../../input/english_words_alpha_370k.txt"):
    if not english_words:
        english_words = get_english_words(path=english_words_file)
        
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        cleaned_paragraph = remove_non_existing_words_from_wordlist(wordlist=paragraph, english_words=english_words)
        cleaned_paragraphs.append(cleaned_paragraph)
        
    return cleaned_paragraphs



def lemmatise_paragraphs(paragraphs, POStag, NLP_MAX_LENGTH=1500000):
    """
    --> function that lemmatises the paragraphs of a single text file.
    """
    
    nlp.max_length = NLP_MAX_LENGTH
    
    #Checks if valid part-of-speech tag was provided
    POStags=["PROPN", "AUX", "NOUN", "ADJ", "VERB", "ADP", "SYM", "NUM"]
    if not isinstance(POStag, str) or POStag.upper() not in POStags:
        raise Exception(f'POSfilter only allows any of the following (SpaCy) part-of-speech tags: {POStags}.')
    
    processed_paragraphs = [text for text in tqdm(nlp.pipe(paragraphs, n_process=2, batch_size=1, disable=["ner", "parser"]), desc=f"Lemmatising ({POStag})...",total=len(paragraphs), leave=False)]
    lemmatized_paragraphs = [[word.lemma_ for word in paragraph if word.pos_ == POStag and not word.is_punct and not word.is_stop] for paragraph in processed_paragraphs]
    regexed_paragraphs= [[re.sub(r'\W+', '', word) for word in paragraph] for paragraph in lemmatized_paragraphs]
   
    return regexed_paragraphs



def lemmatise_city_pair(df, POS, OVERWRITE=False, ONLY_ENGLISH_WORDS=False, ENGLISH_WORDS = [],
    english_words_file="../../../input/english_words_alpha_370k.txt", NLP_MAX_LENGTH=1500000):
    
    for tag in tqdm(POS, desc=f"POS: {POS}", leave=False):
        if OVERWRITE or tag not in df.columns:
            df[f"{tag}"] = lemmatise_paragraphs(paragraphs=df['paragraph'], POStag=tag, NLP_MAX_LENGTH=NLP_MAX_LENGTH)

        if ONLY_ENGLISH_WORDS and (OVERWRITE or f'{tag}_clean' not in df.columns):
            df[f'{tag}_clean'] = keep_english_words_in_paragraphs(paragraphs=df[tag], english_words=ENGLISH_WORDS)
            
    return df



def lemmatise(INPUT_DIR, POS, BATCHES=[], LEMMATISATION_TYPE='', ONLY_ENGLISH_WORDS=False, english_words_file="../../../input/english_words_alpha_370k.txt", OVERWRITE=False, NLP_MAX_LENGTH=1500000):   
    BATCHES = [str(batch) for batch in BATCHES]
    
    #Checks if valid part-of-speech tag was provided
    POStags = ["PROPN", "AUX", "NOUN", "ADJ", "VERB", "ADP", "SYM", "NUM"]
    if not isinstance(POS, list) or len([tag.upper() for tag in POS if tag not in POStags]):
        raise Exception(f'POSfilter only allows any of the following (SpaCy) part-of-speech tags: {POStags}.')
    
    if ONLY_ENGLISH_WORDS:
        with open(english_words_file) as word_file:
            ENGLISH_WORDS = set(word.strip().lower() for word in word_file)

    chosen_batches = [batch for batch in os.listdir(INPUT_DIR) if not BATCHES or batch in BATCHES]
    
#     # Where the magic happens
    for batch in tqdm(chosen_batches, desc=f"BATCHES: {BATCHES}"):
        batch_dir = os.path.join(INPUT_DIR, batch)
        
        for citypair in tqdm(os.listdir(batch_dir), desc="City Pair", leave=False):
            citypair_dir = os.path.join(batch_dir, citypair)
            CITY_PAIR = citypair.split('___')[1]

            df_paragraphs_path = f"{citypair_dir}/{CITY_PAIR}.csv"
            
            if os.path.exists(df_paragraphs_path):
                df = pd.read_csv(df_paragraphs_path)
                df = lemmatise_city_pair(df=df, POS=POS, OVERWRITE=OVERWRITE, ONLY_ENGLISH_WORDS=ONLY_ENGLISH_WORDS, ENGLISH_WORDS=ENGLISH_WORDS, NLP_MAX_LENGTH=NLP_MAX_LENGTH)
                df.to_csv(df_paragraphs_path, index=False)
            else:
                print(f"Batch: {batch}, City Pair: '{CITY_PAIR}' has no file at '{df_paragraphs_path}'.")



"""
Example:
    %%time

    INPUT_DIR = "../../../../../data/clean/city_pair_paragraphs3/"
    BATCHES = [5]
    POS = ["NOUN", "VERB"]
    # LEMMATISATION_TYPE = 'quick', 'accurate'
    ONLY_ENGLISH_WORDS = True
    OVERWRITE = True

    df = lemmatise(INPUT_DIR, POS, BATCHES, ONLY_ENGLISH_WORDS=ONLY_ENGLISH_WORDS, OVERWRITE=OVERWRITE)
"""



def import_lemmatised_paragraphs(INPUT_DIR, POS, BATCHES=[], ONLY_ENGLISH_WORDS=False, merged_POS=True, sort_by_paragraphs=False):
    BATCHES = [str(batch) for batch in BATCHES]
    
    POStags = ["PROPN", "AUX", "NOUN", "ADJ", "VERB", "ADP", "SYM", "NUM"]
    if not isinstance(POS, list) or len([tag for tag in POS if tag.upper() not in POStags]):
        raise Exception(f'POSfilter only allows any of the following (SpaCy) part-of-speech tags: {POStags}.')
    
    chosen_batches = [batch for batch in os.listdir(INPUT_DIR) if not BATCHES or batch in BATCHES]
    
    # Where the magic happens
    data_list = []
    missing_POS = dict()
    collected_POS = set()
    
    for batch in tqdm(chosen_batches, desc=f"BATCHES: {BATCHES}"):
        batch_dir = os.path.join(INPUT_DIR, batch)
        
        for citypair in tqdm(os.listdir(batch_dir), desc="City Pair", leave=False):
            citypair_dir = os.path.join(batch_dir, citypair)
            CITY_PAIR = citypair.split('___')[1]

            df_paragraphs_path = f"{citypair_dir}/{CITY_PAIR}.csv"
            df = pd.read_csv(df_paragraphs_path)
            
            sub_df = df[['city_pair', 'paragraph_id', 'paragraph']]

            if merged_POS:
                sub_df['merged_POS'] = [[] for _ in range(df.shape[0])]
            
            combined_POS = None
            for tag in POS:
                if ONLY_ENGLISH_WORDS:
                    column_name = f'{tag}_clean'   
                else:
                    column_name = f'{tag}'
                
                if column_name not in df.columns:
                    if not column_name in missing_POS.keys():
                        missing_POS[column_name] = []
                        
                    missing_POS[column_name].append(CITY_PAIR)
                    
                else:
                    string_to_list = df[column_name].apply(literal_eval)
                    
                    if merged_POS:
                        sub_df['merged_POS'] += string_to_list    
                        
                    else:   
                        sub_df[tag] = string_to_list
                            
                    collected_POS.add(tag)

             
            citypair_dict = {'batch': batch, 'city_pair': CITY_PAIR, 'paragraphs_count': len(df), 'english_words': ONLY_ENGLISH_WORDS, 'collected_POS': collected_POS, 'lemmatized_paragraphs': sub_df}
            data_list.append(citypair_dict)
    
    if sort_by_paragraphs:
        data_list = sorted(data_list, key=lambda k: k['paragraphs_count'], reverse=True)
    
    if len(missing_POS):
        print(f'The following POS tags are missing: {missing_POS}')
    
    return data_list















# def lemmatise_paragraphs(df, OUTPUT_PATH, POS, ONLY_ENGLISH_WORDS=False, ENGLISH_WORD_LIST=[], OVERWRITE=False, NLP_MAX_LENGTH=1500000):
#     """
#     -->
#         function that lemmatises the paragraphs of a single text file.

#         Parameters:
#         -----------
#             FILE_PATH: Str -> input directory path, to the text files
#             FILE_OUTPUT_DIR: Str -> output directory path, where you want to save the .pickle files
#             POS: string (e.g. "NOUN") -> options: (https://spacy.io/usage/spacy-101#annotations-pos-deps)
#             OVERRIDE_OLD_WORDLISTS: Bool -> Whether you want to override existing output files
#             NLP_MAX_LENGTH: Int (default: 1500000) -> Allowed number of characters per file
#     """
    
#     nlp.max_length = NLP_MAX_LENGTH
    
#     #Checks if valid part-of-speech tag was provided
#     POStags=["PROPN", "AUX", "NOUN", "ADJ", "VERB", "ADP", "SYM", "NUM"]
#     if not isinstance(POS, str) or POS.upper() not in POStags:
#         raise Exception(f'POSfilter only allows any of the following (SpaCy) part-of-speech tags: {POStags}.')
    
#     paragraphs_dict = {}
#     if check_path(OUTPUT_PATH, OVERWRITE):
#         processed_paragraphs = [text for text in tqdm(nlp.pipe(df.paragraph, n_process=2, batch_size=1, disable=["ner", "parser"]), desc=f"Lemmatising ({OUTPUT_PATH.split('___')[1]})",total=len(df.paragraph), leave=False)]
#         lemmatized_paragraphs = [[word.lemma_ for word in paragraph if word.pos_ == POS and not word.is_punct and not word.is_stop] for paragraph in processed_paragraphs]
#         regexed_paragraphs= [[re.sub(r'\W+', '', word) for word in paragraph] for paragraph in lemmatized_paragraphs]
        
#         for index, lemmatised_paragraph in enumerate(regexed_paragraphs):
#             paragraphs_dict[df.loc[index].paragraph_id] = lemmatised_paragraph

#         with open(OUTPUT_PATH, 'wb') as fp:
#             pickle.dump(paragraphs_dict, fp)
    
#     filename = os.path.basename(OUTPUT_PATH)
#     CLEAN_PATH = f"{os.path.dirname(OUTPUT_PATH)}/{'_CLEAN.'.join(filename.split('.'))}"

#     if ONLY_ENGLISH_WORDS and check_path(CLEAN_PATH, OVERWRITE):
#         if not paragraphs_dict:
#             with open(OUTPUT_PATH, 'rb') as file_read:
#                     paragraphs_dict = pickle.load(file_read)
                    
#         for paragraph_id in tqdm(paragraphs_dict, desc='Removing non-existent words', leave=False):
#             cleaned_lemmatised_paragraph = remove_non_existing_words(paragraphs_dict[paragraph_id], ENGLISH_WORD_LIST)
#             paragraphs_dict[paragraph_id] = cleaned_lemmatised_paragraph

#         with open(CLEAN_PATH, 'wb') as file_write:
#             pickle.dump(paragraphs_dict, file_write)



# def is_english_word(word, english_words):
#     return word.lower() in english_words



# def remove_non_existing_words(wordlist: list, english_words) -> list:
#     if not len(english_words):
#         raise Exception("The supplied english words list is empty."
#                        )
#     wordset = set(wordlist)
#     non_existent = []
    
#     for word in wordset:
#         if not is_english_word(word, english_words):
#             non_existent.append(word)
            
#     return([word for word in wordlist if word not in non_existent])



# def lemmatise(INPUT_DIR, POS, BATCHES=[], LEMMATISATION_TYPE='', ONLY_ENGLISH_WORDS=False, english_words_file="../../../input/english_words_alpha_370k.txt", OVERWRITE=False):   
#     BATCHES = [int(x) for x in BATCHES]
#     reg_str = 'biggest_cities_([0-9]+)'
    
#     #Checks if valid part-of-speech tag was provided
#     POStags = ["PROPN", "AUX", "NOUN", "ADJ", "VERB", "ADP", "SYM", "NUM"]
#     if not isinstance(POS, list) or len([tag.upper() for tag in POS if tag not in POStags]):
#         raise Exception(f'POSfilter only allows any of the following (SpaCy) part-of-speech tags: {POStags}.')
    
#     if ONLY_ENGLISH_WORDS:
#         with open(english_words_file) as word_file:
#             ENGLISH_WORDS = set(word.strip().lower() for word in word_file)

#     batch_dirs = [os.path.join(INPUT_DIR, batch) for batch in os.listdir(INPUT_DIR) if not BATCHES or int(re.findall(reg_str, batch)[0]) in BATCHES]

#     # Where the magic happens
#     for batch_dir in tqdm(batch_dirs, desc=f"BATCHES: {BATCHES}"):
        
#         for citypair in tqdm(os.listdir(batch_dir), desc="City Pair", leave=False):
#             citypair_dir = os.path.join(batch_dir, citypair)
#             CITY_PAIR = citypair.split('___')[1]

#             df_paragraphs_path = f"{citypair_dir}/{CITY_PAIR}.csv"
#             if os.path.exists(df_paragraphs_path):
#                 df = pd.read_csv(df_paragraphs_path)

#                 for tag in tqdm(POS, desc=f"POS: {POS}", leave=False):
#                     POS_path = f"{citypair_dir}/lemmatisation/{tag}.pickle"
#                     lemmatise_paragraphs(df=df, 
#                                          OUTPUT_PATH=POS_path,
#                                          POS=tag,
#                                          OVERWRITE=OVERWRITE,
#                                          ONLY_ENGLISH_WORDS=ONLY_ENGLISH_WORDS,
#                                          ENGLISH_WORD_LIST = ENGLISH_WORDS,
#                                          NLP_MAX_LENGTH=1500000)



# def import_lemmatised_paragraphs(INPUT_DIR, POS, BATCHES=[], ONLY_ENGLISH_WORDS=False):
#     BATCHES = [int(x) for x in BATCHES]
#     reg_str = 'biggest_cities_([0-9]+)'
    
#     batch_dirs = [batch for batch in os.listdir(INPUT_DIR) if not BATCHES or int(re.findall(reg_str, batch)[0]) in BATCHES]
    
#     data_dict = {}
#     for batch_name in tqdm(batch_dirs, desc=f"BATCHES: {BATCHES}"):
#         batch_dir = os.path.join(INPUT_DIR, batch_name)
        
#         for citypair in tqdm(os.listdir(batch_dir), desc="City Pair", leave=False):
#             citypair_dir = os.path.join(batch_dir, citypair)
#             CITY_PAIR = citypair.split('___')[1]
            
#             paragraphs_count = len(pd.read_csv(f"{citypair_dir}/{CITY_PAIR}.csv"))
#             data_dict[CITY_PAIR] = {'batch': batch_name, 'original_paragraphs': paragraphs_count, 'english_words': ONLY_ENGLISH_WORDS}
            
            
            
#             for tag in POS:
#                 if ONLY_ENGLISH_WORDS:
#                     file_path = f"{citypair_dir}/lemmatisation/{tag}_CLEAN.pickle"
#                 else:
#                     file_path = f"{citypair_dir}/lemmatisation/{tag}_CLEAN.pickle"
                
#                 if os.path.exists(file_path):
#                     with open(file_path, 'rb') as fp:
#                         lemmatised_paragraphs = pickle.load(fp)

#                         data_dict[CITY_PAIR][tag] = lemmatised_paragraphs
    
#     # Check if all lemmatisation files were present
#     missing = {k: [] for k in POS} 
#     for citypair in data_dict.keys():
#         for tag in POS:
#             if tag not in data_dict[citypair]:
#                 missing[tag].append(citypair)
    
#     for k in missing:
#         if len(missing[k]):
#             print(f"The following city pairs have missing '{k}' files: \n--> {missing[k]}\n")
            
#     print(f'\n Getting lemmatised paragraphs for {len(data_dict.keys())} city pairs...')
    
#     return data_dict


# def lemmatise_multiple_files(INPUT_DIR, POS,  OUTPUT_DIR='', OVERWRITE_PROTECTION=True, NLP_MAX_LENGTH=1500000):
#     """
#     -->
#         function that lemmatises the paragraphs of a batch of text files.

#         Parameters:
#         -----------
#             INPUT_DIR: Str -> input directory path, to the text files
#             OUTPUT_DIR: Str -> output directory path (optional), where you want to save the .pickle files
#             POS: list (default = ['NOUN']) -> options: (https://spacy.io/usage/spacy-101#annotations-pos-deps)
#             OVERWRITE_PROTECTION: Bool (default = True) -> Whether you want to override existing output files
#             NLP_MAX_LENGTH: Int (default: 1500000) -> Allowed number of characters per file
#     """
#     for root, dirs, files in tqdm(list(os.walk(INPUT_DIR))):
        
#         for file in tqdm(files, total=len(files)):
#             if file == f"{root.split('___')[1]}.csv":
#                 FILE_PATH = os.path.join(root, file)
#                 FILE_OUTPUT_DIR = root + '/' + 'lemmatisation'

#                 os.makedirs(FILE_OUTPUT_DIR, exist_ok=True)

#                 lemmatise_single_file(FILE_PATH, FILE_OUTPUT_DIR, POS, OVERWRITE_PROTECTION=OVERWRITE_PROTECTION, NLP_MAX_LENGTH=1500000)



# def lemmatise_single_file(FILE_PATH, FILE_OUTPUT_DIR, POS, OVERWRITE_PROTECTION=True, NLP_MAX_LENGTH=1500000):
#     """
#     -->
#         function that lemmatises the paragraphs of a single text file.

#         Parameters:
#         -----------
#             FILE_PATH: Str -> input directory path, to the text files
#             FILE_OUTPUT_DIR: Str -> output directory path, where you want to save the .pickle files
#             POS: string (e.g. "NOUN") -> options: (https://spacy.io/usage/spacy-101#annotations-pos-deps)
#             OVERRIDE_OLD_WORDLISTS: Bool -> Whether you want to override existing output files
#             NLP_MAX_LENGTH: Int (default: 1500000) -> Allowed number of characters per file
#     """
    
#     nlp.max_length = NLP_MAX_LENGTH
#     POStags=["PROPN", "AUX", "NOUN", "ADJ", "VERB", "ADP", "SYM", "NUM"]
    
#     #Checks if valid part-of-speech tag was provided
#     if not isinstance(POS, str) or POS.upper() not in POStags:
#         raise Exception(f'POSfilter only allows any of the following (SpaCy) part-of-speech tags: {POStags}.')
    
#     CITY_PAIR = FILE_PATH.split('___')[1]
#     new_path = os.path.join(FILE_OUTPUT_DIR, f"{POS}__{CITY_PAIR}__.pickle" )
    
#     if not OVERWRITE_PROTECTION or check_path(new_path):
#         df = pd.read_csv(FILE_PATH)
        
#         processed_paragraphs = [text for text in tqdm(nlp.pipe(df.paragraph, n_process=2, batch_size=1, disable=["ner", "parser"]), total=len(df.paragraph))]
#         lemmatized_paragraphs = [[word.lemma_ for word in paragraph if word.pos_ == POS and not word.is_punct and not word.is_stop] for paragraph in processed_paragraphs]
#         regexed_paragraphs= [[re.sub(r'\W+', '', word) for word in paragraph] for paragraph in lemmatized_paragraphs]
        
#         tempdict = {}
#         for index, lemmatised_paragraph in enumerate(regexed_paragraphs):
#             tempdict[df.loc[index].paragraph_id] = lemmatised_paragraph

#         with open(new_path, 'wb') as fp:
#             pickle.dump(tempdict, fp)



# def import_lemmatised_wordlists(PATH, POS,  BATCHES=[]):

#     """
#     -->
#         function that imports (POS specific) wordlists belonging to specific city pairs.

#         Parameters:
#         -----------
#             PATH: str -> path to lemmatised wordlists (e.g. '../../../../data/enwiki_city_pairs_lemmatised/NOUN/')
#             sort: bool (default = True) -> sort based on the number of city pair co-occurences
#     """
    
#     if not os.path.isdir(PATH):
#         raise Exception("Path is incorrect.")

#     POStags=["PROPN", "AUX", "NOUN", "ADJ", "VERB", "ADP", "SYM", "NUM"]
    
#     #Checks if valid part-of-speech list was provided
#     for tag in POS:
#         if not isinstance(POS, list) or tag.upper() not in POStags:
#             raise Exception(f'POSfilter only allows a list with one or multiple from the following tags: {POStags}.')
    
#     BATCHES = [int(x) for x in BATCHES]
#     datadict= {}
    
#     for root, dirs, files in os.walk(PATH, topdown=True):
#         for name in files:
#             reg_str = 'biggest_cities_([0-9]+)'
#             parent_dir = int(re.findall(reg_str, root)[0])
            
#             if not BATCHES or parent_dir in BATCHES:
#                 CITY_PAIR = root.split('___')[1]
                
#                 if CITY_PAIR not in datadict.keys():
#                     datadict[CITY_PAIR] = {'batch': parent_dir, 'paragraphs': None}
                
#                 for tag in POS:
#                     if name.startswith(tag): #)any(tag for tag in POS in name):
                        
#                         file_path = os.path.join(root, name)
                        
#                         with open(file_path, 'rb') as fp:
#                             lemmatised_paragraphs = pickle.load(fp)

#                             if datadict[CITY_PAIR]['paragraphs'] is None:
#                                 datadict[CITY_PAIR]['paragraphs'] = len(lemmatised_paragraphs.keys())

#                             datadict[CITY_PAIR][tag] = lemmatised_paragraphs
                        
#     # Check if all lemmatisation files were present
#     missing = {k: [] for k in POS} 
#     for citypair in datadict.keys():
#         for tag in POS:
#             if tag not in datadict[citypair]:
#                 missing[tag].append(citypair)
    
#     for k in missing:
#         if len(missing[k]):
#             print(f"The following city pairs have missing '{k}' files: \n--> {missing[k]}\n")
            
#     print(f'\n Getting lemmatised paragraphs for {len(datadict.keys())} city pairs...')
    
#     return datadict



# def lemmatise_files_paragraphs(INPUT_DIR, OUTPUT_DIR, POS, OVERRIDE_OLD_WORDLISTS, NLP_MAX_LENGTH=1500000):
#     """
#     -->
#         function that lemmatises the paragraphs of a batch of text files.

#         Parameters:
#         -----------
#             INPUT_DIR: Str -> input directory path, to the text files
#             OUTPUT_DIR: Str -> output directory path, where you want to save the .pickle files
#             POS: list (default = ['NOUN']) -> options: (https://spacy.io/usage/spacy-101#annotations-pos-deps)
#             OVERRIDE_OLD_WORDLISTS: Bool -> Whether you want to override existing output files
#             NLP_MAX_LENGTH: Int (default: 1500000) -> Allowed number of characters per file
#     """
#     if not os.path.exists(INPUT_DIR):
#         raise Exception("Invalid INPUT Directory Path.")

#     if not os.path.exists(OUTPUT_DIR):
#         raise Exception("Invalid OUTPUT Directory Path.")

#     for root, dirs, files in tqdm(list(os.walk(INPUT_DIR))):

#         # Create subdirectories in output path
#         [os.makedirs(os.path.join(OUTPUT_DIR, dir), exist_ok=True) for dir in dirs]

#         for file in tqdm(files, total=len(files)):
#             file_path = os.path.join(root, file)
#             file_output_dir = root.replace(INPUT_DIR, OUTPUT_DIR)
#             lemmatise_file_paragraphs(FILE_PATH=file_path, FILE_OUTPUT_DIR=file_output_dir, POS=POS, OVERRIDE_OLD_WORDLIST=OVERRIDE_OLD_WORDLISTS, NLP_MAX_LENGTH=NLP_MAX_LENGTH)



# def lemmatise_file_paragraphs(FILE_PATH, FILE_OUTPUT_DIR, POS, OVERRIDE_OLD_WORDLIST, NLP_MAX_LENGTH=1500000):
#     """
#     -->
#         function that lemmatises the paragraphs of a single text file.

#         Parameters:
#         -----------
#             FILE_PATH: Str -> input directory path, to the text files
#             FILE_OUTPUT_DIR: Str -> output directory path, where you want to save the .pickle files
#             POS: list (default = ['NOUN']) -> options: (https://spacy.io/usage/spacy-101#annotations-pos-deps)
#             OVERRIDE_OLD_WORDLISTS: Bool -> Whether you want to override existing output files
#             NLP_MAX_LENGTH: Int (default: 1500000) -> Allowed number of characters per file
#     """

#     nlp.max_length = NLP_MAX_LENGTH
#     POStags=["PROPN", "AUX", "NOUN", "ADJ", "VERB", "ADP", "SYM", "NUM"]

#     #Checks if valid part-of-speech list was provided
#     if not isinstance(POS, list):
#         raise Exception("POS needs to be a list!")

#     for tag in POS:
#         if tag.upper() not in POStags:
#             raise Exception(f'POSfilter only allows a list with one or multiple from the following tags: {POStags}.')

#     CITY_PAIR = os.path.basename(FILE_PATH)[:-4]
#     new_path = os.path.join(FILE_OUTPUT_DIR, f"{''.join(POS)}__{CITY_PAIR}__.pickle" )

#     if OVERRIDE_OLD_WORDLIST or not os.path.exists(new_path):
#         with open(FILE_PATH, 'r', encoding='utf-16') as f:
#             city_pair_paragraphs = [x.strip().lower() for x in f.read().replace('"', "'").replace('“', "'").replace('”', "'").split('\n') if len(x) and 'title=' not in x]
        
#         processed_paragraphs = [text for text in tqdm(nlp.pipe(city_pair_paragraphs, n_process=2, batch_size=1, disable=["ner", "parser"]), total=len(city_pair_paragraphs))]
#         lemmatized_paragraphs = [[word.lemma_ for word in paragraph if word.pos_ in POS and not word.is_punct and not word.is_stop] for paragraph in processed_paragraphs]
#         regexed_paragraphs= [[re.sub(r'\W+', '', word) for word in paragraph] for paragraph in lemmatized_paragraphs]
        
#         with open(new_path, 'wb') as fp:
#             pickle.dump(regexed_paragraphs, fp)



# def import_lemmatised_wordlists(PATH, sort=True):
#     """
#     -->
#         function that imports (POS specific) wordlists belonging to specific city pairs.

#         Parameters:
#         -----------
#             PATH: str -> path to lemmatised wordlists (e.g. '../../../../data/enwiki_city_pairs_lemmatised/NOUN/')
#             sort: bool (default = True) -> sort based on the number of city pair co-occurences
#     """
    
#     if not os.path.isdir(PATH):
#         raise Exception("Path is incorrect.")

#     data = []

#     for root, dirs, files in os.walk(PATH, topdown=True):
#         for name in files:
#             file_path = os.path.join(root, name)
#             parent_dir = os.path.basename(os.path.dirname(file_path))

#             with open(file_path, 'rb') as fp:
#                 data.append((pickle.load(fp), parent_dir, name.split('__')[1]))
    
#     # Sort by number of city pair co-occurences
#     if sort:
#         data = sorted(data, key=lambda x: len(x[0]), reverse=True)

#     return data



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
        
        print('old:', len(wordlist), 'new:', len(cleaned_wordlist), 'removed:', len(wordlist)-len(cleaned_wordlist))
