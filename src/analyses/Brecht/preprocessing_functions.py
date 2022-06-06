import re
import pandas as pd
import os
import time
from tqdm.notebook import tqdm
import unidecode
import numpy as np

# function to read all files from a directory
def read_stream(indir):
    """
    Function to read all the files in a directory (or nested directories)
    containing wikidump extracts.
    Returns a list of strings each element representing the entire contents
    of a single file

        Parameters:
        -----------
        indir (str): path to a directory containing text files
                     or directories with text files

    """
    wikidump = []
    t0 = time.time()

    for root, dirs, files in os.walk(indir):

        for filename in files:
            if not filename.startswith("."):
                fp = os.path.join(root, filename)

                with open(fp, 'r') as f:
                    wikidump.append(f.read())

    t1 = time.time()

    total = t1-t0
    print(f"It took {total}s to read {indir}.")

    return wikidump

# define functions for extracting metadata
def find_title(input_string):
    """function that returns the title of an article"""
    reg_str = "title=\"" + "(.*?)" + "\""
    title = re.findall(reg_str, input_string)
    return title[0]

def find_id(input_string):
    """function that returns the id of an article"""
    reg_str = "id=\"" + "(.*?)" + "\""
    article_id = re.findall(reg_str, input_string)
    return article_id[0]

def find_url(input_string):
    """function that returns the url of an article"""
    reg_str = "url=\"" + "(.*?)" + "\""
    url = re.findall(reg_str, input_string)
    return url[0]

def find_article(input_string):
    """function that returns the body of an article"""

    everything_str = r".*?"
    reg_str = "<doc id=\"" + everything_str + "\">([\S\s]*?)<\/do"

    article = re.findall(reg_str, input_string)

    return article[0]


# function that checks if key words in corpus
def list_in_corpus(list_of_words, text_corpus, n = 2):
    """
    function that checks whether n number of words from a key word list occur in a given corpus.

        Parameters:
        -----------
            list_of_words: (list, str) contains the key words to be matched
            text_corpus:   (str)
            n:             (int, optional)
                minimum number of words from list_of_words that must appear in
                text_corpus. default is 2.
    """
    inclusion = False
    count = 0
    for word in list_of_words:
        if count < n: # only compare if inclusion condition has not yet been met
            if word in text_corpus:
                count += 1 # add 1 every time a city name is in the corpus
        else:
            pass
    if count >= n:
        inclusion = True
        # thus the corpus is only marked for inclusion if
        # at least n cities from the list have been mentioned
    return inclusion


# function to split dumps into flat list

def split_dump(input_dump, split_pattern = "c>"):
    """
    splits list of wikidump documents into a flat list of articles

        Parameters:
        -----------
            input_dump:    a list of strings
            split_pattern: str, optional
                string pattern at which the strings
                should be split into articles. default = 'c>'

    """

    article_list = [
        article for dump
        in tqdm(input_dump, total = len(input_dump), desc = "Progress split_dump()")
        for article in dump.split(split_pattern)]

    return article_list

# function to extract metadata from articles
def process_dump(dump, key_words, message = True):
    """extracts titles and ids from articles containing key words and returns as a list"""

    articles = []
    for article in tqdm(dump, total = len(dump), desc = "Progress"):
        article = unidecode.unidecode(article)
        if (list_in_corpus(key_words, article)):
            try:
                article_id = find_id(article)
                title = find_title(article)
                articles.append((article_id, title, article))
            except:
                pass
        else:
            pass

    if message:
        print(f"After processing {len(articles)} articles remain, "
              f"that is {round(((len(articles)/len(dump))*100), 2)}% "
              f"of the total number of articles ({len(dump)}) in this dump.")

    return articles

def process_dump2(dump, key_words, message = True):
    """extracts titles and ids from articles containing key words and returns as a list"""

    articles = []
    for article in tqdm(dump, total = len(dump), desc = "Progress process_dump()"):
        article = unidecode.unidecode(article)
        if (list_in_corpus(key_words, article)):
            try:
                article_id = find_id(article)
                title = find_title(article)
                article_body = find_article(article)
                articles.append((article_id, title, article_body))
            except:
                pass
        else:
            pass

    if message:
        print(f"{round(((len(articles)/len(dump))*100), 2)}% of articles contain 2 toponyms")

              # f"After processing {len(articles)} articles remain, "
              # f"that is {round(((len(articles)/len(dump))*100), 2)}% "
              # f"of the total number of articles ({len(dump)}) in this dump."
    return articles


# def write_outputcsv(df, outputfp, overwrite_protection = True):
#     """
#     function that writes a dataframe to a csv file. With added overwrite protection
#     """
#     # if the overwrite_protection variable is True a warning message will be displayed
#     # and give the user the option to abort
#     if overwrite_protection:
#         if os.path.exists(outputfp):
#             print(f"File {outputfp} already exists.")
#             print("Are you sure you want to continue and overwrite the file?")
#             decision = input('Continue? [y/n]')
#             if decision == 'y':
#                 df.to_csv(outputfp, index = False)
#                 print(f"file has been written to: {outputfp}")
#
#                 if loop:
#                     print("Do you want to perform the same action for the other files in the directory?")
#                     loop_decision = input('Overwrite subsequent files? [y/n]')
#                     if loop_decision == 'y':
#                         print("Overwrite protection has been turned off.")
#                         overwrite_protection = False
#                     else:
#                         print("Overwrite protection remains on.")
#                 return overwrite_protection
#
#             elif decision == 'n':
#                 print("The process has been halted.")
#                 return
#             else:
#                 print("You did not enter a valid option.")
#                 return
#         else:
#             df.to_csv(outputfp, index = False)
#             print(f"file has been written to: {outputfp}")
#             return
#
#     else:
#         df.to_csv(outputfp, index = False)
#         print(f"file has been written to: {outputfp}")
#
#     return

def write_outputcsv(df, outputfp, overwrite_protection = True):
    # if the overwrite_protection variable is True a warning message will be displayed
    # and give the user the option to abort
    if overwrite_protection:
        if os.path.exists(outputfp):
            print(f"File {outputfp} already exists.")
            print("Are you sure you want to continue and overwrite the file?")
            decision = input('Continue? [y/n]')

            if decision == 'y':
                df.to_csv(outputfp, index = False)
                print(f"file has been written to: {outputfp}")
            elif decision == 'n':
                print("The process has been halted.")
            else:
                print("You did not enter a valid option.\nThe process has halted.")
        else:
            df.to_csv(outputfp, index = False)
            print(f"file has been written to: {outputfp}")

    else:
        df.to_csv(outputfp, index = False)
        print(f"file has been written to: {outputfp}")

    return

##############
# function integrating the other functions
def preprocess(base_dir, outdir, language, key_words, remove_referral=True, overwrite_protection=True):
    """
        params:
            base_dir:             str;
                path to directory where extracted wikidump files can be found
            outdir:               str;
                path where processed files will be saved to (one file per multistream)
            language:             str;
                one of the following ['en', 'fr']
            key_words:            str, list;
                list of strings which must be included in article
            remove_referral:      bool, optional; default is True.
                if True referral pages will be removed
            overwrite_protection: bool, optional; default is True.
                if True confirmation will be asked before overwriting files
    """

    # establish that a valid language was chosen, if not abort function:
    lang_list = ['fr', 'en']
    if language not in lang_list:
        print(f"Invalid language was chosen. \n Please choose one of the following: {lang_list}")
        return

    # creating an output directory
    outdir = os.path.join(outdir, f'{language}wiki/')
#     outdir = f'../../../data/{language}wiki/'

    if not os.path.exists(outdir):
        os.mkdir(outdir)
        print(f'created directory at: {outdir}')
    else:
        pass

    # base input directory
#     base_dir = f"/Volumes/NIJMAN/THESIS/{language}wiki_extracted" # path/to/wikidump/extracted

    # list of multistream directories in base_dir
    dir_list = os.listdir(base_dir)


#     for directory in dir_list:
    for directory in tqdm(dir_list, total = len(dir_list), desc = "Progress Total"):
        dir_fp = os.path.join(base_dir, directory)
        if not directory.startswith("."):
            print(f"\nStarting preprocessing on: {dir_fp}")
            wikidump = read_stream(dir_fp) # read the files in the directory

            wikidump = split_dump(wikidump) # split the files
            wikidump = process_dump2(wikidump, key_words) # extract id, title, article

            df = pd.DataFrame(wikidump, columns = ['article_id', 'title', 'text'])

            if remove_referral:
                try:
                    df['length'] = [len(text.split()) for text in df.text]
                    df['length_title'] = [len(title.split()) for title in df.title]
                    n_referral = len(df[df.length == df.length_title])

                    df = df[['article_id', 'title', 'text']][df.length != df.length_title]
                    print(f"Removing {n_referral} referral pages")

                except:
                    print(f"Referral pages were not removed from multistream {directory}")
                    pass


            # saving the output

            outfile = f'{language}wikidump_{directory}.csv'
            outputfp = os.path.join(outdir, outfile)

            # call write_outputcsv function
            write_outputcsv(df, outputfp, overwrite_protection = overwrite_protection)
        else:
            print(f"Skipping: {dir_fp}")

    print(f"----------\nFiles in {base_dir} have been processed\n----------")

    return

# ---------------------
# matrix generation related functions
def create_city_dict(city_list):
    """
    function that creates a dictionary of name variants to the standard form
    output: a dictionary where the keys are variant names and the values are
    standard names.
    """

    # instantiate dictionary
    city_dict = dict()

    # split up the city names in the city list where a '-' occurs
    # (the symbol used to split separate placenames)
    for city in city_list:
        keys = city.split('-')
        if len(keys) > 1:
            keys.append(city)
        for key in keys:
            city_dict[key] = city

    return city_dict

def city_matrix(city_list):
    """generates an empty matrix with the index/columns consisting of the city names"""

    # create zero matrix with the correct dimensions
    matrix = np.zeros((len(city_list), len(city_list)))

    # transform into dataframe with the columns and index set to the list of cities
    matrix = pd.DataFrame(matrix, columns = city_list)
    matrix['index'] = city_list
    matrix.set_index('index', inplace = True)

    return matrix


def city_appearance(text, dictionary):
    """function to check which placenames appear in the input text per paragraph"""

    # instantiate empty list of standardised city names and city name variations
    cities_variants = []
    cities_standard = []

    # for each word in the text check if the word is a key word in the dictionary(one of the variants)
    for word in dictionary:
        pattern = r"\b" + word + r"\b" #add word boundaries to dictionary word
        match = re.search(pattern, text)
        if match:
            cities_variants.append(word)

    # for each word in the variant replace name with the standard form
    for city in cities_variants:
        city_standard = city.replace(city, dictionary[city])
        cities_standard.append(city_standard)

    return cities_variants, cities_standard

def process_article(article, dictionary, matrix):
    """IMPROVE DOC string
    function that processes each article in order to update the co-occurence values
    in a co-occurence matrix"""

    # split article into paragraphs (by using '\n' as end of paragraph)
    paragraphs = article.splitlines()
    for paragraph in paragraphs:
        # if paragraph empty skip
        if not paragraph:
            continue

        # generate list of cities that appear in the paragraph
        cities_variants, cities_standard = city_appearance(paragraph, dictionary)

        # skip if fewer than 2 cities appear
        if len(set(cities_standard)) < 2:
            continue

        else:
            # create the co-occurences that appear
            for city_i in cities_standard:
                for city_j in cities_standard:
                    if city_i != city_j: # make sure cities don't co-occure with themselves
                        matrix.at[city_i, city_j] += 1 # update value in matrix

    return matrix

def process_corpus(corpus, city_list):
    """function that processes the entire corpus and creates co-occurence matrix"""

    # generate dictionary and matrix
    dictionary = create_city_dict(city_list)
    matrix = city_matrix(city_list)

    # loop over each article in the corpus and update the matrix
    for article in tqdm(corpus, total = len(corpus), desc = "Articles processed"):
        process_article(article, dictionary, matrix)

    return matrix


def create_citylink(matrix):
    """function that creates a dictionary of city pairs and their co-occurence value
    based on code by Tongjing Wang"""

    city_link = {}
    for i in range(len(matrix)-1):
        for j in range(i+1, len(matrix)-1):
            city_link[(matrix.index[i], matrix.columns[j])] = matrix.iloc[i,j]
    return city_link

def write_matrix(matrix, outdir, filename):
    """function to write matrix to csv"""
    outfp = os.path.join(outdir, filename)

    if os.path.exists(outfp):
        print(f"File {outfp} already exists.")
        print("Are you sure you want to continue and overwrite the file?")
        decision = input('Continue? [y/n]')
        if decision == 'y':
            matrix.to_csv(outfp, index = True)
            print(f"Matrix has been written to: {outfp}")
        elif decision == 'n':
            print("The process has been halted.")
        else:
            print("You did not enter a valid option.\nThe process has halted.")
    else:
        matrix.to_csv(outfp, index = True)
        print(f"Matrix has been written to: {outfp}")

    return
