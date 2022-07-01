import os
from tqdm.notebook import tqdm
from scipy.spatial.distance import cosine
from pprint import pprint
import numpy as np


def load_glove_word_embeddings(GLOVE_PATH='.../.../.../.../../glove.42B.300d.txt'):
    """
    --> Function that loads glove word embeddings.

        Parameters:
        -----------
            GLOVE_PATH: Str -> Path to the GloVe file

    """
    if not os.path.exists(GLOVE_PATH):
        raise Exception("The given PATH to the GloVe file doesn't exist.")
    
    
    embeddings_dict = {}
    discarded_dict = {}
    print('This will take approximately ~ 4 minutes...')

    num_lines = sum(1 for line in open(GLOVE_PATH,'r', encoding="utf-8"))

    with open(GLOVE_PATH, 'r', encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines):
            values = line.split()
            token = values[0]
            try:
                vector = np.asarray(values[1:], "float32")
                if vector.shape[0] == 300:
                    embeddings_dict[token] = vector
                else:
                    discarded_dict[token] = vector
            except:
                discarded_dict[token] = None
    
    return embeddings_dict, discarded_dict



def categorize_text(lemmatized_wordlist, mean_vectors_dict, keywords, embeddings_dict, bottom_threshold=0.1, verbose1=False, verbose2=False):
    """
    --> Function that loads glove word embeddings.

        Parameters:
        -----------
            lemmatized_words: List -> List of words.
            keywords: nested list -> List of lists of keywords that represent categories.
            number_of_keywords: Int (default = 1) -> number of keywords to use from a category (setting it to 0 will use them all!)
            bottom_threshold: Float (default = 0.1) -> Lowest allowed similarity value between a word and dominant category.
            verbose1: Bool (default = False) -> Shows similarity calculations between a word and each keyword.
            verbose2: Bool (default = False) -> Shows similarity calculations between a word and each category.

    """
    similar_categories = []
    
    for word in lemmatized_wordlist:
        try:
            word_vector = embeddings_dict[word]
        except:
            continue
            
        if verbose1 or verbose2:
            print(f"word: \t\t'{word}'")

        closeness = []

        for category in keywords:
            try:
                keyword_vector = mean_vectors_dict[category]
            except:
                keyword_vector = embeddings_dict[category]
                
            similarity = 1 - cosine(keyword_vector, word_vector)

            closeness.append((similarity, category))

            if verbose1:
                # print('___________________________')
                print('===>', '\t\tcategory:', category, '\n\t\tsimilarity:', similarity, f"\n")


        similar_category = max(closeness)

        sortedcat = sorted(closeness, key=lambda item: item[0], reverse=True)
        if (sortedcat[0][0] - 0.05) > sortedcat[2][0]:
            allowed = True
        else:
            allowed = False



        if similar_category[0] > bottom_threshold and allowed:
            similar_categories.append((word, similar_category))
            if verbose2:
                #print('category similarity:')
                # pprint(sorted(closeness, key=lambda x: x[0], reverse=True))
                print(f"choice: \tkept")
                print(f"\n==> \tcategory:, {similar_category[1]}, \n\tsimilarity score: {similar_category[0]}")
        elif verbose2:
            print(f"choice: \tdiscarded")
            reason = 'ambiguity' if not allowed else 'low similarity score'
            print(f"reasoning: \t{reason}")

        if verbose2:
            # print(f"analysis: {'not' if not allowed else ''} enough difference\nscores:")
            print("\nscores:\n\t----category----          ----score----")
            for i in sortedcat:
                print(f"\t{i[1]:<10s} \t\t{i[0]}")
            print()

            print('='*100)
            print('')
    
    categories_dict = {key: 0 for key in keywords}

    for x in similar_categories:
        categories_dict[x[1][1]] += x[1][0] #print(x[1])
    

    nonsorted_results = list(sorted(categories_dict.items(), key=lambda item: item[0], reverse=False))
    results = list(sorted(categories_dict.items(), key=lambda item: item[1], reverse=True))
    #pprint(results)

    # print(f"\nThe dominant category is: '{results[0][0]}'", end='')
    #if (results[0][1] - (float(results[0][1])/5)) <= results[1][1]:  
    #    print(f", closely followed by: '{results[1][0]}'.")
    if verbose2:
        print('\n')
        pprint(similar_categories)
    # print('\n --------------------------------------------------------------------')
    
    prediction_dict = {'category_similarities': nonsorted_results, 'prediction': results[0][0]} 

    return prediction_dict



# def categorize_text(lemmatized_wordlist, mean_vectors_dict, keywords, embeddings_dict, number_of_keywords=1, bottom_threshold=0.1, verbose1=False, verbose2=False):
#     """
#     --> Function that loads glove word embeddings.

#         Parameters:
#         -----------
#             lemmatized_words: List -> List of words.
#             keywords: nested list -> List of lists of keywords that represent categories.
#             number_of_keywords: Int (default = 1) -> number of keywords to use from a category (setting it to 0 will use them all!)
#             bottom_threshold: Float (default = 0.1) -> Lowest allowed similarity value between a word and dominant category.
#             verbose1: Bool (default = False) -> Shows similarity calculations between a word and each keyword.
#             verbose2: Bool (default = False) -> Shows similarity calculations between a word and each category.

#     """
#     keywords_to_use = number_of_keywords if number_of_keywords else 1000
#     similar_categories = []
    
#     for word in lemmatized_wordlist:
#         try:
#             word_vector = embeddings_dict[word]
#         except:
#             continue
            
#         if verbose1 or verbose2:
#             print(f"word: \t\t'{word}'")

#         closeness = []

#         for category in keywords:
#             summed_similarity = 0
#             for keyword in category[:keywords_to_use]:
#                 try:
#                     keyword_vector = mean_vectors_dict[keyword]
#                     # print(keyword)
# #                 if isinstance(keyword, np.ndarray):
# #                     keyword = 'Diplomacy'
#                     # keyword_vector = mean_embedding
#                 except:
#                     keyword_vector = embeddings_dict[keyword]
                
#                 value = 1 - cosine(keyword_vector, word_vector)
#                 summed_similarity += value
            
#             normalized_similarity = summed_similarity/len(category[:keywords_to_use])
#             # print(type(normalized_similarity), normalized_similarity)
#             closeness.append((normalized_similarity, category[0]))

#             if verbose1:
#                 # print('___________________________')
#                 print('===>', '\t\tcategory:', category[0], '\n\t\tsimilarity:', normalized_similarity, f"\n\t\tkeywords: {category[:keywords_to_use]}\n")


#         similar_category = max(closeness)

#         sortedcat = sorted(closeness, key=lambda item: item[0], reverse=True)
#         if (sortedcat[0][0] - 0.05) > sortedcat[2][0]:
#             allowed = True
#         else:
#             allowed = False



#         if similar_category[0] > bottom_threshold and allowed:
#             similar_categories.append((word, similar_category))
#             if verbose2:
#                 #print('category similarity:')
#                 # pprint(sorted(closeness, key=lambda x: x[0], reverse=True))
#                 print(f"choice: \tkept")
#                 print(f"\n==> \tcategory:, {similar_category[1]}, \n\tsimilarity score: {similar_category[0]}")
#         elif verbose2:
#             print(f"choice: \tdiscarded")
#             reason = 'ambiguity' if not allowed else 'low similarity score'
#             print(f"reasoning: \t{reason}")

#         if verbose2:
#             # print(f"analysis: {'not' if not allowed else ''} enough difference\nscores:")
#             print("\nscores:\n\t----category----          ----score----")
#             for i in sortedcat:
#                 print(f"\t{i[1]:<10s} \t\t{i[0]}")
#             print()

#             print('='*100)
#             print('')
    
#     categories_dict = {key[0]: 0 for key in descriptive_keywords}
    
#     for x in similar_categories:
#         categories_dict[x[1][1]] += x[1][0] #print(x[1])
    

#     nonsorted_results = list(sorted(categories_dict.items(), key=lambda item: item[0], reverse=False))
#     results = list(sorted(categories_dict.items(), key=lambda item: item[1], reverse=True))
#     #pprint(results)

#     # print(f"\nThe dominant category is: '{results[0][0]}'", end='')
#     #if (results[0][1] - (float(results[0][1])/5)) <= results[1][1]:  
#     #    print(f", closely followed by: '{results[1][0]}'.")
#     if verbose2:
#         print('\n')
#         pprint(similar_categories)
#     # print('\n --------------------------------------------------------------------')
    
#     prediction_dict = {'category_similarities': nonsorted_results, 'prediction': results[0][0]} 

#     return prediction_dict




# def categorize_text(lemmatized_wordlist, keywords, embeddings_dict, number_of_keywords=1, bottom_threshold=0.1, verbose1=False, verbose2=False):
#     """
#     --> Function that loads glove word embeddings.

#         Parameters:
#         -----------
#             lemmatized_words: List -> List of words.
#             keywords: nested list -> List of lists of keywords that represent categories.
#             number_of_keywords: Int (default = 1) -> number of keywords to use from a category (setting it to 0 will use them all!)
#             bottom_threshold: Float (default = 0.1) -> Lowest allowed similarity value between a word and dominant category.
#             verbose1: Bool (default = False) -> Shows similarity calculations between a word and each keyword.
#             verbose2: Bool (default = False) -> Shows similarity calculations between a word and each category.

#     """
#     keywords_to_use = number_of_keywords if number_of_keywords else 1000
#     similar_categories = []

#     for word in lemmatized_wordlist:
#         try:
#             word_vector = embeddings_dict[word]
#         except:
#             continue
            
#         if verbose1 or verbose2:
#             print(f"word: \t\t'{word}'")

#         closeness = []

#         for category in keywords:
#             summed_similarity = 0
#             for keyword in category[:keywords_to_use]:
#                 keyword_vector = embeddings_dict[keyword]
#                 value = 1 - cosine(keyword_vector, word_vector)
#                 summed_similarity += value
            
#             normalized_similarity = summed_similarity/len(category[:keywords_to_use])
#             closeness.append((normalized_similarity, category[0]))

#             if verbose1:
#                 # print('___________________________')
#                 print('===>', '\t\tcategory:', category[0], '\n\t\tsimilarity:', normalized_similarity, f"\n\t\tkeywords: {category[:keywords_to_use]}\n")


#         similar_category = max(closeness)

#         sortedcat = sorted(closeness, key=lambda item: item[0], reverse=True)
#         if (sortedcat[0][0] - 0.05) > sortedcat[2][0]:
#             allowed = True
#         else:
#             allowed = False



#         if similar_category[0] > bottom_threshold and allowed:
#             similar_categories.append((word, similar_category))
#             if verbose2:
#                 #print('category similarity:')
#                 # pprint(sorted(closeness, key=lambda x: x[0], reverse=True))
#                 print(f"choice: \tkept")
#                 print(f"\n==> \tcategory:, {similar_category[1]}, \n\tsimilarity score: {similar_category[0]}")
#         elif verbose2:
#             print(f"choice: \tdiscarded")
#             reason = 'ambiguity' if not allowed else 'low similarity score'
#             print(f"reasoning: \t{reason}")

#         if verbose2:
#             # print(f"analysis: {'not' if not allowed else ''} enough difference\nscores:")
#             print("\nscores:\n\t----category----          ----score----")
#             for i in sortedcat:
#                 print(f"\t{i[1]:<10s} \t\t{i[0]}")
#             print()

#             print('='*100)
#             print('')
    
#     categories_dict = {key[0]: 0 for key in descriptive_keywords}
    
#     for x in similar_categories:
#         categories_dict[x[1][1]] += x[1][0] #print(x[1])
    

#     results = list(sorted(categories_dict.items(), key=lambda item: item[1], reverse=True))
#     pprint(results)

#     print(f"\nThe dominant category is: '{results[0][0]}'", end='')
#     if (results[0][1] - (float(results[0][1])/5)) <= results[1][1]:  
#         print(f", closely followed by: '{results[1][0]}'.")
#     if verbose2:
#         print('\n')
#         pprint(similar_categories)
#     print('\n --------------------------------------------------------------------')
    
#     prediction_dict = {'category_similarities': results, 'prediction': results[0][0]} 

#     return prediction_dict



def categorize_group_of_texts(lemmatized_wordlists, keywords, embeddings_dict, number_of_keywords=1, bottom_threshold=0.1, verbose1=False, verbose2=False):
    categories = {}
    for index, lemmatized_wordlist in tqdm(enumerate(lemmatized_wordlists), total=len(lemmatized_wordlists)):
        prediction_dict = categorize_text(lemmatized_wordlist=lemmatized_wordlist, keywords=keywords, embeddings_dict=embeddings_dict, number_of_keywords=number_of_keywords, bottom_threshold=bottom_threshold, verbose1=verbose1, verbose2=verbose2)
        try:
            categories[prediction_dict['prediction']] +=1
        except:
            categories[prediction_dict['prediction']] = 1
    
    return categories


