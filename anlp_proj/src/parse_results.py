import pandas as pd
import os

RESULTS = '/cs/labs/tomhope/guy_korn/ANLP_PROJECT/results'


def get_deepest_dirs(base_dir, target, file_name):
    result = []
    for root, dirs, files in os.walk(base_dir):
        if target in root and not dirs and files:
            rel_path = os.path.relpath(root, start=RESULTS)
            rel_path = os.path.join(rel_path, file_name)
            result.append(rel_path)

    return result


def parse_separate_datasets_results():
    sep_res_paths = get_deepest_dirs(RESULTS, 'separate_datasets', 'knn-acc.txt')
    print(len(sep_res_paths))
    print("\n".join(sep_res_paths))

    rows = []
    for file_path in sep_res_paths:
        components = file_path.split("/")

        with open(os.path.join(RESULTS, file_path), "r") as file:
            accuracy = float(file.read().strip().split(": ")[1])

        rows.append({"section_split": components[1],
                     "model": components[2],
                     "train_type": components[3],
                     "section_pooling": components[5].split("_")[0],
                     "optimized_section": 'optimized' in components[5],
                     "patient_pooling": components[6],
                     "accuracy": accuracy
                     })

    data = pd.DataFrame(rows)
    print(data.iloc[0])
    data.to_csv(RESULTS + '/separate_datasets.csv')

parse_separate_datasets_results()

def parse_comorbidities():
    comor_res_path = get_deepest_dirs(RESULTS, 'comorbidities', 'ons_vs_rest.csv')
    print(len(comor_res_path))
    print("\n".join(comor_res_path))
    dfs = []
    for path in comor_res_path:
        components = path.split("/")
        csv = pd.read_csv(os.path.join(RESULTS, path)).set_index('base classifier')
        csv = csv.stack().reset_index()
        csv.columns = ['clf', 'metric', 'value']
        csv = csv.assign(**{"section_split": components[1],
                            "model": components[2],
                            "train_type": components[3],
                            "section_pooling": components[5].split("_")[0],
                            "optimized_section": 'optimized' in components[5],
                            "patient_pooling": components[6],
                            })
        dfs.append(csv)
    data = pd.concat(dfs)
    print(data.iloc[0])
    data.to_csv(RESULTS + '/commodities.csv')
parse_comorbidities()
