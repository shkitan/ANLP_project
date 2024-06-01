import json
import os
import pandas as pd
from bs4 import BeautifulSoup


def get_xml_soup(xml_path):
    with open(xml_path, 'r') as f:
        data = f.read()
    soup = BeautifulSoup(data, "xml")
    return soup


def read_xml_to_df(xml_path):
    soup = get_xml_soup(xml_path)
    docs = []
    for doc in soup.find_all('doc'):
        docs.append({'id': doc['id'], 'text': doc.text.strip()})

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
    return value >= 100


def get_cat_names(file_name):
    file = read_xml_to_df(file_name)
    lst_obj = file['text'].to_list()
    cat_names_freq_dict = {}
    for text in lst_obj:
        lines = text.split('\n')
        cat_name = ''
        for line in lines[2:]:
            if ':' in line and not any(char.isdigit() for char in
                                       line):
                cat_name = line.split(':')[0]
                if cat_name not in cat_names_freq_dict:
                    cat_names_freq_dict[cat_name] = 1
                else:
                    cat_names_freq_dict[cat_name] += 1
    return dict(filter(filter_cat, cat_names_freq_dict.items())).keys(), file


s = set()


def process_file(file, cat_names):
    lst_obj = file['text'].to_list()
    id_index = 0
    dict_obj = {}
    for text in lst_obj:
        id = file['id'].get(id_index)
        if id in s:
            print("errorrrrr shkitan doble id")
        s.add(id)
        id_index += 1
        dict_cat_text = obj_process(text, cat_names)
        dict_obj[id] = dict_cat_text
    return dict_obj


def iterate_dirs():
    dict_file_d = {}
    for filename in os.scandir('obesity_challenge'):
        if filename.is_file():
            cat_names, file = get_cat_names(filename.path)
            dict_obj = process_file(file, cat_names)
            dict_file_d[filename.name] = dict_obj
            print(filename.name)

    with open('obs.json', 'w') as file:
        json.dump(dict_file_d, file)


iterate_dirs()
