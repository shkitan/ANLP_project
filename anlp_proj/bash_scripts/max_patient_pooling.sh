#!/bin/bash


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


#obesity
train_or_test="train"
mean_or_cls="mean"
section_embedding_dir="embedding/obs-max/${train_or_test}/${model_name}/${finetuned}/sections/${mean_or_cls}"
cmd="python patient_pooling.py --section_embedding_dir ${section_embedding_dir}"
echo "${cmd}"
eval "${cmd}"


train_or_test="test"
mean_or_cls="mean"
section_embedding_dir="embedding/obs-max/${train_or_test}/${model_name}/${finetuned}/sections/${mean_or_cls}"
cmd="python patient_pooling.py --section_embedding_dir ${section_embedding_dir}"
echo "${cmd}"
eval "${cmd}"


train_or_test="train"
mean_or_cls="cls"
section_embedding_dir="embedding/obs-max/${train_or_test}/${model_name}/${finetuned}/sections/${mean_or_cls}"
cmd="python patient_pooling.py --section_embedding_dir ${section_embedding_dir}"
echo "${cmd}"
eval "${cmd}"


train_or_test="test"
mean_or_cls="cls"
section_embedding_dir="embedding/obs-max/${train_or_test}/${model_name}/${finetuned}/sections/${mean_or_cls}"
cmd="python patient_pooling.py --section_embedding_dir ${section_embedding_dir}"
echo "${cmd}"
eval "${cmd}"


#smokers
train_or_test="train"
mean_or_cls="mean"
section_embedding_dir="embedding/smokers-max/${train_or_test}/${model_name}/${finetuned}/sections/${mean_or_cls}"
cmd="python patient_pooling.py --section_embedding_dir ${section_embedding_dir}"
echo "${cmd}"
eval "${cmd}"


train_or_test="test"
mean_or_cls="mean"
section_embedding_dir="embedding/smokers-max/${train_or_test}/${model_name}/${finetuned}/sections/${mean_or_cls}"
cmd="python patient_pooling.py --section_embedding_dir ${section_embedding_dir}"
echo "${cmd}"
eval "${cmd}"


train_or_test="train"
mean_or_cls="cls"
section_embedding_dir="embedding/smokers-max/${train_or_test}/${model_name}/${finetuned}/sections/${mean_or_cls}"
cmd="python patient_pooling.py --section_embedding_dir ${section_embedding_dir}"
echo "${cmd}"
eval "${cmd}"


train_or_test="test"
mean_or_cls="cls"
section_embedding_dir="embedding/smokers-max/${train_or_test}/${model_name}/${finetuned}/sections/${mean_or_cls}"
cmd="python patient_pooling.py --section_embedding_dir ${section_embedding_dir}"
echo "${cmd}"
eval "${cmd}"