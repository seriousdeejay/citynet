# Main Script

import os
#import bs4
import requests
import pathlib
import wikipediaapi

## print(os.getcwd())

file_location = os.path.realpath(__file__)
cwd = os.path.dirname(file_location)
os.chdir(cwd)

european_capitals = '../input/city_pages.txt'
CAPITAL_PAGES_OUTPUT_FOLDER = '../input/european-capital-pages/'
european_capital_pages = []


def download_european_capital_pages():
    """ Download wikipedia pages of european capitals
    """
    with open(european_capitals, "r", encoding="utf-8") as f:
        for i, data in enumerate(f.readlines()):
            if len(data.strip()) and data[0] != '#':
                european_capital_pages.append(data.replace("\n", ""))

    ## print(european_capital_pages)

    # Access wikipedia API
    wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

    pathlib.Path(CAPITAL_PAGES_OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    count = 0
    for link in european_capital_pages:
        city_name = link.split('/')[-1]
        p_wiki = wiki_wiki.page(city_name)
        file_name = f"{CAPITAL_PAGES_OUTPUT_FOLDER}/{city_name}.txt"
        if not os.path.exists(file_name):
            with open(file_name, "w", encoding='utf-8') as txt_file:
                txt_file.write(p_wiki.text)
            count += 1
    
    print(f"Added {count} pages.")

# download_european_capital_pages()


def load_european_capital_pages():
    """ Load wikipedia pages of european capitals
    """
    capital_pages = {}
    page_paths =  os.listdir(CAPITAL_PAGES_OUTPUT_FOLDER)
    for page_path in page_paths:
        with open(f"{CAPITAL_PAGES_OUTPUT_FOLDER}{page_path}", "r", encoding="utf-8") as f:
            capital_pages[page_path.replace('.txt','')] = [line.rstrip() for line in f]      

    ## print(capital_pages['Amsterdam'])

# load_european_capital_pages()




""" Scraper template with the use of requests & beautifulsoup
"""
# response = requests.get("https://en.wikipedia.org/wiki/Python_(programming_language)")

# if response is not None:
#     page = bs4.BeautifulSoup(response.text, 'html.parser')

#     title = page.select('#firstHeading')[0].text
#     paragraphs = page.select('p')
    
#     print(title)
#     intro = '\n'.join([ paragraph.text for paragraph in paragraphs[0:5]])
#     print(intro)

