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

cmd="python section_embedding.py --model_name ${model_name} --finetuned ${finetuned} --dataset_path datasets/obesity/obs_train.json"
echo "${cmd}"
eval "${cmd}"

cmd="python section_embedding.py --model_name ${model_name} --finetuned ${finetuned} --dataset_path datasets/obesity/obs_test.json"
echo "${cmd}"
eval "${cmd}"

cmd="python section_embedding.py --model_name ${model_name} --finetuned ${finetuned} --dataset_path datasets/smokers/smokers_train.json"
echo "${cmd}"
eval "${cmd}"

cmd="python section_embedding.py --model_name ${model_name} --finetuned ${finetuned} --dataset_path datasets/smokers/smokers_test.json"
echo "${cmd}"
eval "${cmd}"


