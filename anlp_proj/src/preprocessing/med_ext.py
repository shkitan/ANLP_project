import json
import os
import pandas as pd
import tarfile


def get_xml_soup(xml_path):
    tar = tarfile.open(xml_path, "r:gz")
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is not None:
            content = f.read()
def read_xml_to_df(xml_path):
    soup = get_xml_soup(xml_path)
    docs = []
    for doc in soup.find_all('RECORD'):
        docs.append({'id': doc['ID'], 'text': str(doc.TEXT)})

    # Create a pandas DataFrame from the extracted data
    df = pd.DataFrame(docs)
    return df

def is_word_uppercase(word):
    for char in word:
        if char == " ":
            continue
        if not char.isupper():
            return False
    return True
def obj_process(text, cat_names):
    lines = text.split('\n')
    dict_cat_text = {}
    # dict_cat_text['meta_data'] = lines[0]
    meata = True
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


def filter_cat(pair):
    key, value = pair
    return value >= 30

def get_cat_names():
    cat_names_freq_dict = {}
    for dir in os.scandir('medication extraction'):
        for filename in os.scandir(dir):
            f = open(filename, "r")
            # open file
            text = f.read()
            lines = text.split('\n')
            cat_name = ''
            meata = True
            for line in lines[1:]:
                if not (':' in line and not any(char.isdigit() for char in
                                                line)) and  meata:
                    continue
                if ':' in line and not any(char.isdigit() for char in
                                           line):
                    meata = False
                    cat_name = line.split(':')[0]
                    if cat_name not in cat_names_freq_dict:
                        cat_names_freq_dict[cat_name] = 1
                    else:
                        cat_names_freq_dict[cat_name] += 1
    return dict(filter(filter_cat, cat_names_freq_dict.items())).keys()

def process_file(cat_names):
    dict_file_d = {}
    cat_names_freq_dict = {}
    dict_obj = {}
    for dir in os.scandir('medication extraction'):
        for filename in os.scandir(dir):
            f = open(filename, "r")
            # open file
            text = f.read()
            dict_cat_text = obj_process(text, cat_names)
            dict_obj[filename] = dict_cat_text
    return dict_obj


def iterate_dirs():
    cat_names = get_cat_names()
    dict_obj = process_file(cat_names)
    with open('med_ext.json', 'w') as file:
        json.dump(dict_obj, file)


iterate_dirs()