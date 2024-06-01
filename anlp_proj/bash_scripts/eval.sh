!/bin/bash


while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model_name)
            model_name="$2"
            shift 2
            ;;
        --finetuned)
            finetuned="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$model_name" ] || [ -z "$finetuned" ]; then
    echo "Error: model_name and finetuned are required."
    usage
    exit 1
fi


export PYTHONPATH=./


# eval comorbidities Discharge Pretrain model, sectioned by Semantic, section pooling by mean, patient pooling mean and pca

mean_or_cls="mean"
pca_or_mean="pca"
train_embedding_dir="embedding/obs/train/${model_name}/${finetuned}/patient/${mean_or_cls}_sections/${pca_or_mean}"
test_embedding_dir="embedding/obs/test/${model_name}/${finetuned}/patient/${mean_or_cls}_sections/${pca_or_mean}"
cmd="python evaluation_tasks/downstream_comorbidities.py --embeddings_train ${train_embedding_dir} --embeddings_test ${test_embedding_dir}"
echo "${cmd}"
eval "${cmd}"


mean_or_cls="cls"
pca_or_mean="pca"
train_embedding_dir="embedding/obs/train/${model_name}/${finetuned}/patient/${mean_or_cls}_sections/${pca_or_mean}"
test_embedding_dir="embedding/obs/test/${model_name}/${finetuned}/patient/${mean_or_cls}_sections/${pca_or_mean}"
cmd="python evaluation_tasks/downstream_comorbidities.py --embeddings_train ${train_embedding_dir} --embeddings_test ${test_embedding_dir}"
echo "${cmd}"
eval "${cmd}"


mean_or_cls="mean"
pca_or_mean="mean"
train_embedding_dir="embedding/obs/train/${model_name}/${finetuned}/patient/${mean_or_cls}_sections/${pca_or_mean}"
test_embedding_dir="embedding/obs/test/${model_name}/${finetuned}/patient/${mean_or_cls}_sections/${pca_or_mean}"
cmd="python evaluation_tasks/downstream_comorbidities.py --embeddings_train ${train_embedding_dir} --embeddings_test ${test_embedding_dir}"
echo "${cmd}"
eval "${cmd}"


mean_or_cls="cls"
pca_or_mean="mean"
train_embedding_dir="embedding/obs/train/${model_name}/${finetuned}/patient/${mean_or_cls}_sections/${pca_or_mean}"
test_embedding_dir="embedding/obs/test/${model_name}/${finetuned}/patient/${mean_or_cls}_sections/${pca_or_mean}"
cmd="python evaluation_tasks/downstream_comorbidities.py --embeddings_train ${train_embedding_dir} --embeddings_test ${test_embedding_dir}"
echo "${cmd}"
eval "${cmd}"


# eval ds separation Discharge Pretrain model, sectioned by Semantic, section pooling by mean, patient pooling mean and pca
section_pooling="mean"
patient_pooling="pca"
cmd="python evaluation_tasks/seperate_datasets.py --obs_train  embedding/obs/train/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling}  --obs_test  embedding/obs/test/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling} --smokers_train embedding/smokers/train/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling} --smokers_test embedding/smokers/test/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling}"
echo "${cmd}"
eval "${cmd}"


section_pooling="mean"
patient_pooling="mean"
cmd="python evaluation_tasks/seperate_datasets.py --obs_train  embedding/obs/train/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling}  --obs_test  embedding/obs/test/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling} --smokers_train embedding/smokers/train/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling} --smokers_test embedding/smokers/test/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling}"
echo "${cmd}"
eval "${cmd}"


section_pooling="cls"
patient_pooling="pca"
cmd="python evaluation_tasks/seperate_datasets.py --obs_train  embedding/obs/train/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling}  --obs_test  embedding/obs/test/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling} --smokers_train embedding/smokers/train/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling} --smokers_test embedding/smokers/test/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling}"
echo "${cmd}"
eval "${cmd}"


section_pooling="cls"
patient_pooling="mean"
cmd="python evaluation_tasks/seperate_datasets.py --obs_train  embedding/obs/train/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling}  --obs_test  embedding/obs/test/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling} --smokers_train embedding/smokers/train/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling} --smokers_test embedding/smokers/test/${model_name}/${finetuned}/patient/${section_pooling}_sections/${patient_pooling}"
echo "${cmd}"
eval "${cmd}"
