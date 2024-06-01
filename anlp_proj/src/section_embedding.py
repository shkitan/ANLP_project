import json
import ast
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

from config import EMBEDDING_DIR
from load_data import get_records_by_patient
from load_models import load_model

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def extract_ordered_sections_texts(section_dict):  # TODO
    """
    Our data is built such that the section_dict has keys in the form of "section_name-section_idx"
    where section_idx is the order in which the section appeared on the discharge note of the patient.
    Keeping section texts ordered is important for label alignment in downstreams jobs.
    """

    # todo: uncomment after bug fix.
    # # Define a custom key function to extract the integer from the key name
    # key_func = lambda key: int(key.split('-')[-1])
    def key_func(key):
        if '-' in key:
            return int(key.split('-')[-1])
        else:
            return float('inf')  # Set a very large number for keys without '-number'

    # Sort the keys based on the integer values
    sorted_keys = sorted(section_dict.keys(), key=key_func)

    # Extract the values based on the sorted keys
    sorted_values = [section_dict[key] for key in sorted_keys]
    return sorted_values


### Pooling : Tokens (matrix) -> Section (vector) ###
def pool_tokens_vectors(vectors, tokenizer, tokens, pooling_method):
    """
    Map section embeddings to a single vector representing a section.
    This can be mean pooling, taking the first [CLS] token as embedding,
    attention-based pooling, or transformer-based pooling.
    """
    if pooling_method == "mean":
        # Apply mean pooling to the vectors
        pooled_vector = torch.mean(vectors, dim=0)

    # Apply the mean pooling only on embeddings vectors of the CLS tokens which were trained for classifications
    elif pooling_method == "cls":
        # Find the indices of [CLS] tokens
        cls_indices = [i for i, token in enumerate(tokens) if token == tokenizer.cls_token_id]

        # Convert the indices to a tensor with dtype torch.int64
        cls_indices_tensor = torch.tensor(cls_indices, dtype=torch.int64)

        # Extract the embeddings of [CLS] tokens
        cls_embeddings = torch.index_select(vectors, dim=0, index=cls_indices_tensor)

        # Calculate the mean of [CLS] token embeddings
        pooled_vector = torch.mean(cls_embeddings, dim=0)
    else:
        raise ValueError("Invalid pooling method specified.")
    assert sum(pooled_vector.isnan()).item() == 0
    return pooled_vector


# Process the text and extract embeddings for each section with sliding window
def embed_patient_sections(sections_dict, tokenizer, model, max_window_size):
    """
    Given the sections of a single instance (section_name-index in patient-> text), embed all tokens in it,
    and use pooling method to create a single vector for every section.
    If a section is longer than max_window_size, use a sliding window to embed all tokens in it and use pooling method
    on all vectors of the section.
    returns a tensor shaped (num_of_sections, emb_dim)
    """
    sections_texts = extract_ordered_sections_texts(sections_dict)
    cls_embeddings = []
    mean_embeddings = []
    for text in sections_texts:
        # Tokenize the section
        # todo: check if this make sense (why not tokenizer(text)?)
        tokens = tokenizer.encode(text.strip(), add_special_tokens=True, truncation=False)
        section_length = len(tokens)

        # Apply sliding window to process long sections
        if section_length > max_window_size:
            # Initialize starting index and window size
            start_index = 0
            window_size = max_window_size
            current_section_embeddings = []

            while start_index < section_length:
                # Adjust window size if it exceeds section length
                if start_index + window_size > section_length:
                    window_size = section_length - start_index

                # Extract tokens within the sliding window
                window_tokens = tokens[start_index: start_index + window_size]
                # Convert tokens to PyTorch tensors
                input_ids = torch.tensor(window_tokens).unsqueeze(0).to(device)
                # Forward pass through the model
                outputs = model(input_ids)
                # Extract the token embeddings from the last hidden state
                embeddings = outputs.last_hidden_state.squeeze(0)
                # Add to the list of current section embeddings
                current_section_embeddings.append(embeddings)
                # Slide the window
                start_index += window_size
            current_section_embeddings = torch.cat(current_section_embeddings, dim=0)

        else:
            # Convert tokens to PyTorch tensors
            input_ids = torch.tensor(tokens).unsqueeze(0).to(device)

            # Forward pass through the model
            outputs = model(input_ids)

            # Extract the token embeddings from the last hidden state
            current_section_embeddings = outputs.last_hidden_state.squeeze(0)

        # Pool the token embeddings
        current_section_embeddings = current_section_embeddings.to('cpu')
        cls_pooled_embedding = pool_tokens_vectors(current_section_embeddings, tokenizer, tokens, 'cls')
        mean_pooled_embedding = pool_tokens_vectors(current_section_embeddings, tokenizer, tokens, 'mean')

        cls_embeddings.append(cls_pooled_embedding)
        mean_embeddings.append(mean_pooled_embedding)

    return cls_embeddings, mean_embeddings


# Function to process each instance in the JSON file
def process_instance(instance, tokenizer, model, output_dir, max_window_size, override=True):
    patient_id = instance['patient_id']
    cls_out_path = os.path.join(output_dir / "cls", f"{patient_id}.npy")
    mean_out_path = os.path.join(output_dir / "mean", f"{patient_id}.npy")

    # Process the text and extract embeddings for each section with sliding window
    cls_embeddings, mean_embeddings = embed_patient_sections(instance['sections'], tokenizer, model, max_window_size)

    # Convert section embeddings to NumPy arrays
    cls_embeddings = [embedding.detach().numpy() for embedding in cls_embeddings]
    mean_embeddings = [embedding.detach().numpy() for embedding in mean_embeddings]

    # Save the section embeddings to disk
    np.save(cls_out_path, cls_embeddings)
    np.save(mean_out_path, mean_embeddings)


def run_section_embedding(model, tokenizer, dataset, output_dir: str, max_window_size: int, override=True):
    # Process each model separately
    os.makedirs(output_dir, exist_ok=True)

    # Process each instance in the JSON file for the current model
    pb = tqdm(total=len(dataset))  # progress bar
    cls_out_dir = os.path.join(output_dir / "cls")
    mean_out_dir = os.path.join(output_dir / "mean")
    os.makedirs(cls_out_dir, exist_ok=True)
    os.makedirs(mean_out_dir, exist_ok=True)
    for i, instance in enumerate(dataset):
        process_instance(instance, tokenizer, model, output_dir, max_window_size, override)
        pb.update(1)  # update progress bar
    pb.close()  # close progress bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="model checkpoint, in huggingface or local")
    parser.add_argument("--finetuned", help="whether to use finetune (boolean)")
    parser.add_argument("--dataset_path", default=False, help="ds_name_split(train/test) format.")
    parser.add_argument("--sections_split", default=False, help="semantic/max.")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    is_finetuned = ast.literal_eval(args.finetuned)
    # todo: Not sure if its cool to load finetuned model with AutoModel - need to check architecture differences.
    tokenizer, model = load_model(args.model_name, is_finetuned, args.sections_split)
    dataset = get_records_by_patient(os.path.join(os.getcwd(), args.dataset_path), num_samples=args.num_samples)

    dataset_name, split = args.dataset_path.split("/")[-1].split("_")
    model_type = "finetuned" if is_finetuned else "pretrained"
    # embs / dataset / {train, test} / {BioBERT...} / {pretrained,finetuned} / {sections,opt_sections, patient} / pooling / pat_id.npy
    output_dir = EMBEDDING_DIR / dataset_name / split.split(".")[0] / args.model_name / model_type / "sections"

    run_section_embedding(model, tokenizer, dataset, output_dir, max_window_size=tokenizer.model_max_length, override=True)
