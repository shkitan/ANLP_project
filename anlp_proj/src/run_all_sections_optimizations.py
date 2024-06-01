import subprocess
import argparse
import os
import config

def submit_slurm_job(command, job_name, embeddings_path):
    log_dir = os.path.join(embeddings_path, "logs_optimization")
    os.makedirs(log_dir, exist_ok=True)

    output_log = os.path.join(log_dir, f"{job_name}.out")
    error_log = os.path.join(log_dir, f"{job_name}.err")

    slurm_script = f'''#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=40g
#SBATCH --job-name="{job_name}"
#SBATCH --time=0-1
#SBATCH --output="{output_log}"
#SBATCH --error="{error_log}"

{command}
'''
    with open("slurm_script.sh", "w") as f:
        f.write(slurm_script)

    subprocess.run(["sbatch", "slurm_script.sh"])
    subprocess.run(["rm", "slurm_script.sh"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="BioBERT", help="model checkpoint, in huggingface or local")
    parser.add_argument("--finetuned", default="False", help="True, False, or all")
    parser.add_argument("--embeddings_path", default=config.EMBEDDING_DIR, help="path to embeddings dir")
    parser.add_argument("--data_name", default="obesity", help="obesity or smokers")
    parser.add_argument("--sections_max_split", action="store_true", default=False, help="True if sections are split by max seq len, False if split by semantic meaning")
    parser.add_argument("--dataset_split", default="train", help="train or test or all")
    parser.add_argument("--sections_pooling_method", default='mean', help="mean or cls")
    args = parser.parse_args()

    model_names = {'BioBERT', 'ClinicalBERT', 'ClinicalBERT_Discharge_Summary_BERT'}
    dataset_names = {"obesity", "smokers"}
    pooling_methods = {"mean", "cls"}

    model_name_list = [args.model_name] if args.model_name != "all" else model_names
    dataset_name_list = [args.data_name] if args.data_name != "all" else dataset_names
    sections_pooling_method_list = [args.sections_pooling_method] if args.sections_pooling_method != "all" else pooling_methods

    if args.finetuned == "all":
        finetuned_list = ['True', 'False']
    else:
        finetuned_list = [str(args.finetuned.lower() == "true")]

    
    if args.dataset_split == "all":
        dataset_split_list = ["train", "test"]
    else:
        dataset_split_list = [args.dataset_split]

    for model_name in model_name_list:
        for finetuned in finetuned_list:
            for dataset_name in dataset_name_list:
                for dataset_split in dataset_split_list:
                    for pooling_method in sections_pooling_method_list:
                        max_split_flag = "--sections_max_split" if args.sections_max_split else ""

                        command = f"python run_sections_optimizations.py --model_name {model_name} --finetuned {finetuned} --embeddings_path {args.embeddings_path} --data_name {dataset_name} {max_split_flag} --dataset_split {dataset_split} --sections_pooling_method {pooling_method}"
                        job_name = f"anlp_proj_{model_name}_{finetuned}_{dataset_name}_{dataset_split}_{pooling_method}"
                        # print(f"Submitting job {job_name} with command:\n{command}")
                        # print(command, ';')
                        submit_slurm_job(command, job_name, args.embeddings_path)