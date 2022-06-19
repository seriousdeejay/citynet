import requests
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from wordcloud import WordCloud

from tqdm.notebook import tqdm
from wordfreq import word_frequency
import os
import pickle
from collections import Counter

import pandas as pd



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



def create_wordcloud(data, WORDCLOUD_LOCATION, OVERRIDE_OLD_WORDCLOUDS = False):
    """
    -->
        function that creates wordcloud for a list of lemmatized wordlists.

        Parameters:
        -----------
            data: list of tuples -> Output from 'import_lemmatised_wordlists' (i.e. wordlist, parent_dir, filename (with city pair names)). 
            OVERRIDE_OLD_WORLDCLOUDS: bool (default = False) -> whether you want to override existing wordclouds
            WORDCLOUD_LOCATION: str -> Path to save wordclouds to

    """
    if not os.path.isdir(WORDCLOUD_LOCATION):
        raise Exception("Path is an invalid directory.")
        
    for i in tqdm(range(len(data)), desc='lemmatized_wordlists'):       
        os.makedirs(os.path.join(WORDCLOUD_LOCATION, data[i][1]), exist_ok=True)
        filename = f"{WORDCLOUD_LOCATION}/{data[i][1]}/{data[i][-1]}.png"
        
        if OVERRIDE_OLD_WORDCLOUDS or not os.path.exists(filename):
            
            print(filename)
            wordfreq = Counter(data[i][0])
                
            # remove both cities from wordcloud
            cities = data[i][-1].split('_')
            for city in cities:
                wordfreq[city] = 0
                
            print(f"\r{data[i][2], wordfreq.most_common(4)}", end="                               ")
    
            wordcloud.generate_from_frequencies(wordfreq)
            wordcloud.to_file(filename)
        else:
            print('already exists...')



def create_wordcloud_from_df(df, WORDCLOUD_LOCATION, OVERRIDE_OLD_WORDCLOUDS = False):
    if not os.path.isdir(WORDCLOUD_LOCATION):
        raise Exception("Path is an invalid directory.")
    # i = 0
    for city_pair in tqdm(df.columns, total=len(df.columns), desc='lemmatized_wordlists'):
        # time.sleep(1)
        # city_pair = [df.columns[city_pair]]
        # i += 1
        # print(i, city_pair)
        # continue
        filename = os.path.join(WORDCLOUD_LOCATION,city_pair + '.png')
        
        if OVERRIDE_OLD_WORDCLOUDS or not os.path.exists(filename):
            
            wordfrequency = df[city_pair].to_dict()

            # remove both cities from wordcloud
            cities = city_pair.split('_')
            for city in cities:
                wordfrequency[city] = 0

            print(f"\r{cities, sorted(wordfrequency.items(), key=lambda x: x[1], reverse=True)[:5]}", end="                               ")

            wordcloud.generate_from_frequencies(wordfrequency)
            # print(filename)
            wordcloud.to_file(filename)
        else:        
            print(f"\rwordcloud for {city_pair} already exists.", end="                             ")