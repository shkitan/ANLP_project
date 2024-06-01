#!/bin/bash

export PYTHONPATH=./

./bash_scripts/max_section_embedding.sh --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False
./bash_scripts/max_section_embedding.sh --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned True

./bash_scripts/max_section_embedding.sh --model_name ClinicalBERT --finetuned False
./bash_scripts/max_section_embedding.sh --model_name ClinicalBERT --finetuned True

./bash_scripts/max_section_embedding.sh --model_name BioBERT --finetuned False
./bash_scripts/max_section_embedding.sh --model_name BioBERT --finetuned True


./bash_scripts/max_patient_pooling.sh --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned pretrained
./bash_scripts/max_patient_pooling.sh --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned finetuned

./bash_scripts/max_patient_pooling.sh --model_name ClinicalBERT --finetuned pretrained
./bash_scripts/max_patient_pooling.sh --model_name ClinicalBERT --finetuned finetuned

./bash_scripts/max_patient_pooling.sh --model_name BioBERT --finetuned pretrained
./bash_scripts/max_patient_pooling.sh --model_name BioBERT --finetuned finetuned


./bash_scripts/eval.sh --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned pretrained
./bash_scripts/eval.sh --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned finetuned

./bash_scripts/eval.sh --model_name ClinicalBERT --finetuned pretrained
./bash_scripts/eval.sh --model_name ClinicalBERT --finetuned finetuned

./bash_scripts/eval.sh --model_name BioBERT --finetuned pretrained
./bash_scripts/eval.sh --model_name BioBERT --finetuned finetuned