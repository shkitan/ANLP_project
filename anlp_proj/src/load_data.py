import json
import numpy as np
import os
from itertools import chain
from datasets import Dataset, concatenate_datasets

MLM_SEMANTIC_PATHS = ["datasets/obesity/obs_train.json",
                      "datasets/smokers/smokers_train.json",
                      "datasets/meds_extract/med_ext.json"]
MLM_MAXIMAL_PATHS = ["datasets/obesity/obs-max_train.json",
                     "datasets/smokers/smokers-max_train.json",
                     "datasets/meds_extract/med_ext_max.json"]

EMBEDDING_SEMANTIC_DS_PATHS = ["datasets/obesity/obs_train.json",
                               "datasets/smokers/smokers_train.json",
                               "datasets/obesity/obs_test.json",
                               "datasets/smokers/smokers_train.json",
                               ]

EMBEDDING_MAXIMAL_DS_PATHS = ["datasets/obesity/obs-max_train.json",
                              "datasets/smokers/smokers-max_train.json",
                              "datasets/obesity/obs-max_test.json",
                              "datasets/smokers/smokers_randon_train.json",
                              ]

def dummy_data():
    # Example dictionary of patient IDs and sections
    data = {"1": {"section_1": "Text of section 1 for patient 1.",
                  "section_2": "Text of section 2 for patient 1.",
                  "section_3": "Text of section 3 for patient 1."},
            "2": {"section_1": "Text of section 1 for patient 2.",
                  "section_2": "Text of section 2 for patient 2."},
            }
    return data


def parse_patient_sections(data, num_samples):
    # We want that each patient-section will be a record. with text.
    records = []
    for idx, (patient_id, sections) in enumerate(data.items()):
        if num_samples is not None and idx >= num_samples:
            break
        for section_name, section_text in sections.items():
            record = {"patient_id": patient_id,
                      "section_id": section_name,
                      "text": section_text,
                      }
            records.append(record)
    return records


def get_dummy_dataset_by_sections():
    data = dummy_data()
    data = parse_patient_sections(data, None)
    return Dataset.from_list(data)


def get_dataset_by_sections(path, num_samples=None):
    # in the format of patient_id -> sections
    # section is dict with section_name -> text.
    with open(path, 'r') as file:
        data = json.load(file)
    data = parse_patient_sections(data, num_samples)
    return data

def get_mlm_dataset(section_split: str, num_samples: int = None):
    if section_split == 'semantic':
        ds = [get_dataset_by_sections(os.path.join(os.getcwd(), path), num_samples) for path in MLM_SEMANTIC_PATHS]
        ds = list(chain.from_iterable(ds)) # flatten list
        return Dataset.from_list(ds)
    elif section_split == 'max':
        ds = [get_dataset_by_sections(os.path.join(os.getcwd(), path), num_samples) for path in MLM_MAXIMAL_PATHS]
        ds = list(chain.from_iterable(ds))  # flatten list
        return Dataset.from_list(ds)
    else:
        raise Exception("Only max or semantic are valid arguments.")

def get_records_by_patient(path, num_samples=None):
    # in the format of patient_id -> sections
    # section is dict with section_name -> text.
    with open(path, 'r') as file:
        data = json.load(file)
    records = []
    for idx, (pid, sections) in enumerate(data.items()):
        if num_samples is not None and idx >= num_samples:
            break
        record = {'patient_id': pid, 'sections': sections}
        records.append(record)
    return records


def get_patients_dict(path, num_samples=None):
    with open(path, 'r') as file:
        data = json.load(file)
    if num_samples is not None:
        pids = np.random.choice(data.keys(), num_samples)
        data = {pid: data[pid] for pid in pids}
    return data

def load_npy_folder(folder_path):
    matrix_dict = {}
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is in npy format
        if file_name.endswith('.npy'):
            # Load the npy file as a numpy array
            matrix = np.load(file_path)

            # Append the matrix to the dict
            matrix_dict[file_name.replace(".npy", "")] = matrix
    return matrix_dict

def load_embeddings(emb_dir):
    """ loads the section_embedding.py from path (section_embedding.py were saved with np.save)
    Section level
    # each file in dir correspond to a matrix of [section, embedding_dim]
    # this function will return tensor of [num_patient, section, embedding_dim]
    Patient level - matirx [patients, embedding_dim]
    """
    # # for sanity check, randomly generate samples. comment this when we have real embeddings
    # n = 10000
    # d = 10
    # embeddings_array = np.random.normal(loc=0, scale=1, size=d * n).reshape(n, d)

    return load_npy_folder(emb_dir)
