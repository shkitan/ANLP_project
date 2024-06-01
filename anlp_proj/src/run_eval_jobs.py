import os
import subprocess

PROJECT_ROOT = '/cs/labs/tomhope/guy_korn/ANLP_PROJECT/'


def submit_slurm_job(command, job_name):
    log_dir = "/cs/labs/tomhope/guy_korn/ANLP_PROJECT/results/logs_optimization"
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


def get_final_dirs_with_files(base_dir, target_dir):
    final_dirs_with_files = []

    for root, dirs, files in os.walk(base_dir):
        if target_dir in root and not dirs and files:
            rel_path = os.path.relpath(root, start=PROJECT_ROOT)
            final_dirs_with_files.append(rel_path)

    return final_dirs_with_files


# Example usage:
main_directory = os.path.join(PROJECT_ROOT, 'embedding')
final_dirs_with_files = get_final_dirs_with_files(main_directory, target_dir='patient')

# convert to train-test tuples
result = []
for p in final_dirs_with_files:
    if 'train/' in p:
        result.append((p, p.replace('train/', 'test/')))
    elif 'test/' in p:
        result.append((p.replace('test/', 'train/'), p))
    else:
        print("Error")

print("Len tuples with duplicates: ", len(result))
result = list(set(result))
print("Len without duplicates: ", len(result))

comor_cmds = []
for p in result:
    if '/obs' in p[0]:
        obs_embedding_train, obs_embedding_test = p
        sec_split = 'max' if "-max" in obs_embedding_train else 'semantic'
        comorbidities_cmd = f"python evaluation_tasks/downstream_comorbidities.py --section_split {sec_split} --embeddings_train  {obs_embedding_train} --embeddings_test  {obs_embedding_test}"
        submit_slurm_job(comorbidities_cmd, 'commor_eval')
        # comor_cmds.append(comorbidities_cmd)
#
# print(len(comor_cmds))
# print(";\n".join(comor_cmds))

# sep_cmds = []
# for p in result:
#     if '/obs' in p[0]:
#         obs_train, obs_test = p
#         smoke_train, smoke_test = obs_train.replace('/obs', '/smokers'), obs_test.replace('/obs', '/smokers')
#         sec_split = 'max' if "-max" in obs_train else 'semantic'
#         sep_cmd = f"python evaluation_tasks/seperate_datasets.py --section_split {sec_split} --obs_train {obs_train} --obs_test {obs_test} --smokers_train {smoke_train} --smokers_test {smoke_test}"
#         print("Submit: ", sep_cmd)
#         submit_slurm_job(sep_cmd, 'sep_ds_eval')
#         sep_cmds.append(sep_cmd)
# print("len serparate commands: ", len(sep_cmds))
# print(";\n".join(sep_cmds))
