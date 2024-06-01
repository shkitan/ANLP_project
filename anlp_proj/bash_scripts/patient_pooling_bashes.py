import os


def get_final_dirs_with_files(base_dir, target_dir):
    final_dirs_with_files = []

    for root, dirs, files in os.walk(base_dir):
        if target_dir in root and not dirs and files:
            final_dirs_with_files.append(root)

    return final_dirs_with_files


# Example usage:
main_directory = 'embedding'
final_dirs_with_files = get_final_dirs_with_files(main_directory, target_dir='section')

result = []
for p in final_dirs_with_files:
    if 'figures' in p:
        p = p.replace('/figures', '')
        cmd = f"python patient_pooling.py --section_embedding_dir  {p}"
        result.append(cmd)

print(";\n".join(result))