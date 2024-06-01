import json
import os
import pandas as pd
from bs4 import BeautifulSoup


def get_xml_soup(xml_path):
    """
    Parses an XML file and returns the XML data as a BeautifulSoup object.

    Parameters:
        xml_path (str): The file path of the XML file to be parsed.

    Returns:
        soup (BeautifulSoup object): The parsed XML data as a BeautifulSoup object.

    Dependencies:
        - This function requires the `BeautifulSoup` class from the `bs4` (Beautiful Soup) library.

    Example:
        xml_file_path = 'path/to/xml_file.xml'
        xml_soup = get_xml_soup(xml_file_path)
    """
    with open(xml_path, 'r') as f:
        data = f.read()
    soup = BeautifulSoup(data, "xml")
    return soup

def read_xml_to_df_obs(xml_path):
    """
    Reads an XML file, extracts 'doc' elements, and returns a pandas DataFrame containing the 'id' and 'text' fields.

    Parameters:
        xml_path (str): The file path of the XML file to be read.

    Returns:
        df (pandas DataFrame): A DataFrame containing the extracted 'id' and 'text' fields from the 'doc' elements.

    Dependencies:
        - This function relies on the 'get_xml_soup' function, which must be defined and accessible.

    Example:
        xml_file_path = 'path/to/xml_file.xml'
        df = read_xml_to_df_obs(xml_file_path)
    """
    soup = get_xml_soup(xml_path)
    docs = []
    for doc in soup.find_all('doc'):
        docs.append({'id': doc['id'], 'text': doc.text.strip()})
    df = pd.DataFrame(docs)
    return df
def read_xml_to_df_smokers(xml_path):
    """
    this function used to process the file from smokers dataset
    Reads an XML file, extracts 'RECORD' elements, and returns a pandas DataFrame containing the 'ID' and 'TEXT' fields.

    Parameters:
        xml_path (str): The file path of the XML file to be read.

    Returns:
        df (pandas DataFrame): A DataFrame containing the extracted 'ID' and 'TEXT' fields from the 'RECORD' elements.

    Dependencies:
        - This function relies on the 'get_xml_soup' function, which must be defined and accessible.

    Example:
        xml_file_path = 'path/to/xml_file.xml'
        df = read_xml_to_df_smokers(xml_file_path)
    """
    soup = get_xml_soup(xml_path)
    docs = []
    for doc in soup.find_all('RECORD'):
        docs.append({'id': doc['ID'], 'text': str(doc.TEXT)})

    # Create a pandas DataFrame from the extracted data
    df = pd.DataFrame(docs)
    return df

def is_word_uppercase(word):
    """
    Checks if a word consists entirely of uppercase characters, excluding spaces.

    Parameters:
        word (str): The word to be checked.

    Returns:
        bool: True if the word consists only of uppercase characters, False otherwise.

    Example:
        word = "HELLO"
        is_upper = is_word_uppercase(word)
    """
    for char in word:
        if char == " ":
            continue
        if not char.isupper():
            return False
    return True

def obj_process(text):
    """
    Processes the input text and splits it into categories based on uppercase words followed by a colon.

    Parameters:
        text (str): The input text to be processed.

    Returns:
        dict: A dictionary containing the categories as keys and the corresponding text as values.

    Example:
        text = "META_DATA: This is the meta data.\nDESCRIPTION: This is the description."
        processed_text = obj_process(text)
    """
    lines = text.split('\n')
    dict_cat_text = {}
    text = lines[0]
    cur_cat = 'meta_data'
    for line in lines[1:]:
        if ':' in line and is_word_uppercase(line.split(':')[0]):
            dict_cat_text[cur_cat] = text
            cur_cat = line.split(':')[0]
            text = line.split(':')[1]
        else:
            text += '\n'
            text += line
            continue
    dict_cat_text[cur_cat] = text
    return dict_cat_text

def process_file(file):
    """
    Processes a file containing text and IDs, and creates a dictionary of objects with IDs as keys and processed text as values.

    Parameters:
        file (DataFrame): A DataFrame containing 'text' and 'id' columns.

    Returns:
        dict: A dictionary with IDs as keys and processed text as values.

    Example:
        data = {'text': ['META_DATA: This is the meta data.\nDESCRIPTION: This is the description.',
                         'META_DATA: Another meta data.\nDESCRIPTION: Another description.'],
                'id': [1, 2]}
        df = pd.DataFrame(data)
        processed_data = process_file(df)
    """
    lst_obj = file['text'].to_list()
    id_index = 0
    dict_obj = {}
    s = set()
    for text in lst_obj:
        id = file['id'].get(id_index)
        if id in s:
            print("errorrrrr shkitan doble id")
        s.add(id)
        id_index += 1
        dict_cat_text = obj_process(text)
        dict_obj[id] = dict_cat_text
    return dict_obj

def process_file_med_ext():
    """
    Processes files in the 'medication extraction' directory and its subdirectories, and creates a dictionary of objects with file names as keys and processed text as values.

    Returns:
        dict: A dictionary with file names as keys and processed text as values.

    Example:
        processed_data = process_file_med_ext()
    """
    dict_obj = {}
    for dir in os.scandir('medication extraction'):
        for filename in os.scandir(dir):
            f = open(filename, "r")
            text = f.read()
            dict_cat_text = obj_process(text)
            dict_obj[filename.name] = dict_cat_text
    return dict_obj

def iterate_dirs():
    """
    Iterates through directories and processes files to create JSON files for different challenges.

    - Reads XML files in the 'obesity_challenge' directory and creates a JSON file 'obs.json' containing processed data.
    - Processes files in the 'medication extraction' directory and its subdirectories, and creates a JSON file 'med_ext.json' containing processed data.
    - Reads XML files in the 'smokers_challenge' directory and creates a JSON file 'smokers.json' containing processed data.

    Example:
        iterate_dirs()
    """
    #obs
    dict_file_d = {}
    for filename in os.scandir('obesity_challenge'):
        if filename.is_file():
            file = read_xml_to_df_obs(filename)
            dict_obj = process_file(file)
            dict_file_d[filename.name] = dict_obj

    with open('obs.json', 'w') as file:
        json.dump(dict_file_d, file)

    #med ext
    dict_obj = process_file_med_ext()
    with open('med_ext.json', 'w') as file:
        json.dump(dict_obj, file)

    #smokers
    dict_file_d = {}
    for filename in os.scandir('smokers_challenge'):
        if filename.is_file():
            print(filename.name)
            file = read_xml_to_df_smokers(filename.path)
            dict_obj = process_file(file)
            dict_file_d[filename.name] = dict_obj
    with open('smokers.json', 'w') as file:
        json.dump(dict_file_d, file)

iterate_dirs()