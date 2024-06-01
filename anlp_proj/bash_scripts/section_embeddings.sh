### Semantic
## BioBERT
# Fine tuned
python section_embedding.py --model_name BioBERT --finetuned True --sections_split semantic --dataset_path datasets/obesity/obs_train.json;
python section_embedding.py --model_name BioBERT --finetuned True --sections_split semantic --dataset_path datasets/obesity/obs_test.json;
python section_embedding.py --model_name BioBERT --finetuned True --sections_split semantic --dataset_path datasets/smokers/smokers_train.json;
python section_embedding.py --model_name BioBERT --finetuned True --sections_split semantic --dataset_path datasets/smokers/smokers_test.json
# Pretrained
python section_embedding.py --model_name BioBERT --finetuned False --sections_split semantic --dataset_path datasets/obesity/obs_train.json;
python section_embedding.py --model_name BioBERT --finetuned False --sections_split semantic --dataset_path datasets/obesity/obs_test.json;
python section_embedding.py --model_name BioBERT --finetuned False --sections_split semantic --dataset_path datasets/smokers/smokers_train.json;
python section_embedding.py --model_name BioBERT --finetuned False --sections_split semantic --dataset_path datasets/smokers/smokers_test.json

## Bio Clinical BERT
# Fine tuned
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split semantic --dataset_path datasets/obesity/obs_train.json;
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split semantic --dataset_path datasets/obesity/obs_test.json;
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split semantic --dataset_path datasets/smokers/smokers_train.json;
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split semantic --dataset_path datasets/smokers/smokers_test.json
# Pretrained
python section_embedding.py --model_name ClinicalBERT --finetuned False --sections_split semantic --dataset_path datasets/obesity/obs_train.json;
python section_embedding.py --model_name ClinicalBERT --finetuned False --sections_split semantic --dataset_path datasets/obesity/obs_test.json;
python section_embedding.py --model_name ClinicalBERT --finetuned False --sections_split semantic --dataset_path datasets/smokers/smokers_train.json;
python section_embedding.py --model_name ClinicalBERT --finetuned False --sections_split semantic --dataset_path datasets/smokers/smokers_test.json

## Discharge BioBERT
# Fine tuned
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned True --sections_split semantic --dataset_path datasets/obesity/obs_train.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned True --sections_split semantic --dataset_path datasets/obesity/obs_test.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned True --sections_split semantic --dataset_path datasets/smokers/smokers_train.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned True --sections_split semantic --dataset_path datasets/smokers/smokers_test.json
# Pretrained
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --sections_split semantic --dataset_path datasets/obesity/obs_train.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --sections_split semantic --dataset_path datasets/obesity/obs_test.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --sections_split semantic --dataset_path datasets/smokers/smokers_train.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --sections_split semantic --dataset_path datasets/smokers/smokers_test.json

########!!!!!!!!!!!!!!!###########
### max
## BioBERT
# Fine tuned
python section_embedding.py --model_name BioBERT --finetuned True --sections_split max --dataset_path datasets/obesity/obs-max_train.json;
python section_embedding.py --model_name BioBERT --finetuned True --sections_split max --dataset_path datasets/obesity/obs-max_test.json;
python section_embedding.py --model_name BioBERT --finetuned True --sections_split max --dataset_path datasets/smokers/smokers-max_train.json;
python section_embedding.py --model_name BioBERT --finetuned True --sections_split max --dataset_path datasets/smokers/smokers-max_test.json
# Pretrained
python section_embedding.py --model_name BioBERT --finetuned False --sections_split max --dataset_path datasets/obesity/obs-max_train.json;
python section_embedding.py --model_name BioBERT --finetuned False --sections_split max --dataset_path datasets/obesity/obs-max_test.json;
python section_embedding.py --model_name BioBERT --finetuned False --sections_split max --dataset_path datasets/smokers/smokers-max_train.json;
python section_embedding.py --model_name BioBERT --finetuned False --sections_split max --dataset_path datasets/smokers/smokers-max_test.json

## Bio Clinical BERT
# Fine tuned
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split max --dataset_path datasets/obesity/obs-max_train.json;
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split max --dataset_path datasets/obesity/obs-max_test.json;
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split max --dataset_path datasets/smokers/smokers-max_train.json;
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split max --dataset_path datasets/smokers/smokers-max_test.json
# Pretrained
python section_embedding.py --model_name ClinicalBERT --finetuned False --sections_split max --dataset_path datasets/obesity/obs-max_train.json;
python section_embedding.py --model_name ClinicalBERT --finetuned False --sections_split max --dataset_path datasets/obesity/obs-max_test.json;
python section_embedding.py --model_name ClinicalBERT --finetuned False --sections_split max --dataset_path datasets/smokers/smokers-max_train.json;
python section_embedding.py --model_name ClinicalBERT --finetuned False --sections_split max --dataset_path datasets/smokers/smokers-max_test.json

## Discharge BioBERT
# Fine tuned
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned True --sections_split max --dataset_path datasets/obesity/obs-max_train.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned True --sections_split max --dataset_path datasets/obesity/obs-max_test.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned True --sections_split max --dataset_path datasets/smokers/smokers-max_train.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned True --sections_split max --dataset_path datasets/smokers/smokers-max_test.json
# Pretrained
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --sections_split max --dataset_path datasets/obesity/obs-max_train.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --sections_split max --dataset_path datasets/obesity/obs-max_test.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --sections_split max --dataset_path datasets/smokers/smokers-max_train.json;
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --sections_split max --dataset_path datasets/smokers/smokers-max_test.json
###################
###################
###################

## Optimize Sections
python run_all_sections_optimizations.py --model_name all --finetuned all --data_name all --dataset_split all --sections_pooling_method all --sections_max_split
python run_all_sections_optimizations.py --model_name all --finetuned all --data_name all --dataset_split all --sections_pooling_method all

## Patient Pooling



