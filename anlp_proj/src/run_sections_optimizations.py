import argparse
import ast
import os
from pathlib import Path

#from config import EMBEDDING_DIR, DATASETS_DIR
import config
from load_data import get_patients_dict
from optimize_sections_emb import train_embedding_model
from patient_pooling import run_patient_pooling

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="BioBERT", help="model checkpoint, in huggingface or local")
    parser.add_argument("--finetuned", type=str, default="False", help="whether to use finetuned model or not. Values are True or False.")
    parser.add_argument("--embeddings_path", default=config.EMBEDDING_DIR, help="path to embeddings dir")
    parser.add_argument("--data_name", default="obesity", help="obesity or smokers")
    parser.add_argument("--sections_max_split", action="store_true", default=False, help="True if sections are split by max seq len, False if split by semantic meaning")
    parser.add_argument("--dataset_split", default="train", help="train or test")
    parser.add_argument("--sections_pooling_method", default='mean', help="mean or cls")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--margin", type=float, default=2.0, help="margin for triplet loss")
    parser.add_argument("--positive_weight", type=float, default=1.0, help="weight for the positive example of triplet loss")
    parser.add_argument("--org_dist_weight", type=float, default=0.5, help="weight for the distance of the new embeddings from the original")

    args = parser.parse_args()

    model_type = "finetuned" if args.finetuned.lower() == "true" else "pretrained"
    emb_dataset_name = args.data_name.replace('obesity', 'obs')
    if args.sections_max_split:
        emb_dataset_name += '-max'
    load_dataset_name = config.DATASETS_DIR / f"{args.data_name}/{emb_dataset_name.replace('-max', '')}_{args.dataset_split}.json"
    # if args.embeddings_path is str -> convert to Path
    if isinstance(args.embeddings_path, str):
        args.embeddings_path = Path(args.embeddings_path)

    # embs / dataset / {train, test} / {BioBERT...} / {pretrained,finetuned} / {sections,opt_sections, patient} / pooling / pat_id.npy
    section_embedding_dir = args.embeddings_path / f"{emb_dataset_name}" / args.dataset_split / args.model_name / model_type / "sections" / args.sections_pooling_method

    # map args.data_name to dataset_name
    # obs -> obesity and if sections_max_split is True ->add'-max'


    print(f"Optimizing Sections for\n{args.data_name=} {emb_dataset_name=} {args.dataset_split} {args.model_name} {model_type} {args.sections_pooling_method} sections_split={args.sections_max_split}...")
    opt_section_emb = str(section_embedding_dir).replace('sections', 'optimized_sections')
    patient_data = get_patients_dict(load_dataset_name)
    train_embedding_model(patient_data, section_embedding_dir, opt_section_emb, section_are_max_seq_len=args.sections_max_split, lr=args.lr, epochs=args.epochs, margin=args.margin, positive_weight=args.positive_weight, org_dist_weight=args.org_dist_weight)
    #section_embedding_dir = opt_section_emb