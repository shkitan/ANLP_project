import argparse
from pathlib import Path
from load_data import load_embeddings
from tqdm.auto import tqdm
import os
import numpy as np
from sklearn.decomposition import PCA
import ast

def pca_pooling(sections_embedding: np.array):

    """
    Given a tensor of [patients, sections, embedding_dim], use PCA to get matrix of [patient, embedding_dim]
    :param tensor:
    :return:
    """
    patient_embedding = PCA(n_components=1).fit_transform(sections_embedding.T).squeeze()
    return patient_embedding

def mean_pooling(sections_embedding: np.array):
    return sections_embedding.mean(axis=0)

def get_pool_method(method_name):
    if method_name == "pca":
        return pca_pooling
    elif method_name == "mean":
        return mean_pooling
    else:
        raise Exception("Only pca/mean pooling are available for patients.")

def run_patient_pooling(patient_sections: dict, pooling_method: str, output_dir: str):
    pool = get_pool_method(pooling_method)
    pb = tqdm(total=len(patient_sections))  # progress bar
    for pat_id, section_embedding in patient_sections.items():
        patient_embedding = pool(section_embedding)
        assert np.isnan(patient_embedding).sum() == 0, pat_id
        np.save(f'{output_dir}/{pat_id}', patient_embedding)
        pb.update(1)  # update progress bar
    pb.close()  # close progress bar

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser()
    # Add the arguments
    parser.add_argument("--section_embedding_dir", help="model checkpoint, in huggingface or local")
    parser.add_argument("--debug", default="False")
    # Parse the arguments
    args = parser.parse_args()

    # embs / dataset / {train, test} / {BioBERT...} /{pretrained,finetuned} / {sections,opt_sections,patient} / pooling / pat_id.npy
    section_embedding_dir = Path(args.section_embedding_dir)
    prefix = section_embedding_dir.parent.parent
    section_pool = section_embedding_dir.name # last level on sections embedding
    section_emb_type = section_embedding_dir.parent.name

    output_dir = prefix / "patient" / '_'.join([section_pool, section_emb_type])
    os.makedirs(output_dir / 'mean', exist_ok=True)
    os.makedirs(output_dir / 'pca', exist_ok=True)
    if ast.literal_eval(args.debug):
        patient_sections = {i : np.random.rand(10, 768) for i in range(10)}
    else:
        patient_sections = load_embeddings(section_embedding_dir)

    print("Mean Patient Pooling")
    run_patient_pooling(patient_sections, 'mean', output_dir / 'mean')
    print("PCA Patient Pooling")
    run_patient_pooling(patient_sections, 'pca', output_dir / 'pca')
