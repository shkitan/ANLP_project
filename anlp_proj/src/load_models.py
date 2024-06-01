import os
from transformers import AutoTokenizer, AutoModel
import torch
from config import PRETRAINED_MODELS, FINETUNED_MODELS
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

BERT_MAX_TOKENS = 512
def get_pretrained_model(name, sections_split=None, model_class=AutoModel):
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=PRETRAINED_MODELS)
    tokenizer.model_max_length = 512
    model = model_class.from_pretrained(name, cache_dir=PRETRAINED_MODELS)
    return tokenizer, model

def get_fine_tuned_model(name, sections_split, model_class=AutoModel):
    if sections_split not in ['semantic', 'max']:
        raise Exception("Only semantic/max are valid splits for finetuned model")
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=PRETRAINED_MODELS)
    tokenizer.model_max_length = 512
    finetuned_path = os.path.join(FINETUNED_MODELS, sections_split, name.split("/")[-1])
    model = model_class.from_pretrained(finetuned_path)
    return tokenizer, model

def load_model(model_to_use, finetuned, sections_split=None, model_class=AutoModel):
    """
    Loads every model and tokenizer that their name appear in models_to_use.
    returns a dict where keys are models names, and values are dicts with model iteslf and tokenizer.
    """
    model_getter = get_fine_tuned_model if finetuned else get_pretrained_model
    if model_to_use == "BioBERT":
        tokenizer, model = model_getter("dmis-lab/biobert-base-cased-v1.1", sections_split, model_class)
    elif model_to_use == "ClinicalBERT":
        tokenizer, model = model_getter("emilyalsentzer/Bio_ClinicalBERT", sections_split, model_class)
    elif model_to_use == "ClinicalBERT_Discharge_Summary_BERT":
        tokenizer, model = model_getter("emilyalsentzer/Bio_Discharge_Summary_BERT", sections_split, model_class)
    else:
        print("invalid model name")
        return None
    model.to(device)
    return tokenizer, model