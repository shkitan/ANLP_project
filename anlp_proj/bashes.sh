# Section Embedding for all Datasets sectioned by Semantic, for Discharge Pretrain model
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --dataset_path datasets/obesity/obs_train.json
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --dataset_path datasets/obesity/obs_test.json
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --dataset_path datasets/obesity/smokers_train.json
python section_embedding.py --model_name ClinicalBERT_Discharge_Summary_BERT --finetuned False --dataset_path datasets/obesity/smokers_test.json


# ---- GUY -----
## Section Embedding for all Datasets sectioned by MAX, for BIO BERT
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split semantic --dataset_path datasets/obesity/obs_train.json;
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split semantic --dataset_path datasets/obesity/obs_test.json;
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split semantic --dataset_path datasets/smokers/smokers_train.json;
python section_embedding.py --model_name ClinicalBERT --finetuned True --sections_split semantic --dataset_path datasets/smokers/smokers_test.json

# Patient Pooling (2 variants) for each of the four options.
python patient_pooling.py --section_embedding_dir  embedding/obs/train/ClinicalBERT_Discharge_Summary_BERT/finetuned/sections/mean/;
python patient_pooling.py --section_embedding_dir  embedding/obs/train/ClinicalBERT_Discharge_Summary_BERT/finetuned/sections/cls/;
python patient_pooling.py --section_embedding_dir  embedding/obs/test/ClinicalBERT_Discharge_Summary_BERT/finetuned/sections/mean/;
python patient_pooling.py --section_embedding_dir  embedding/obs/test/ClinicalBERT_Discharge_Summary_BERT/finetuned/sections/cls/;

python patient_pooling.py --section_embedding_dir  embedding/smokers/train/ClinicalBERT_Discharge_Summary_BERT/finetuned/sections/mean/;
python patient_pooling.py --section_embedding_dir  embedding/smokers/train/ClinicalBERT_Discharge_Summary_BERT/finetuned/sections/cls/;
python patient_pooling.py --section_embedding_dir  embedding/smokers/test/ClinicalBERT_Discharge_Summary_BERT/finetuned/sections/mean/;
python patient_pooling.py --section_embedding_dir  embedding/smokers/test/ClinicalBERT_Discharge_Summary_BERT/finetuned/sections/cls/;

# Fine tune all models on max sections
#python fine_tune_mlm.py --model BioBERT --section_split max
#python fine_tune_mlm.py --model ClinicalBERT_Discharge_Summary_BERT --section_split max
#python fine_tune_mlm.py --model ClinicalBERT --section_split max

# eval comorbidities Discharge Pretrain model, sectioned by Semantic, section pooling by mean, patient pooling mean and pca
python evaluation_tasks/downstream_comorbidities.py --embeddings_train  embedding/obs/train/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/pca/ --embeddings_test  embedding/obs/test/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/pca/
python evaluation_tasks/downstream_comorbidities.py --embeddings_train  embedding/obs/train/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/mean/ --embeddings_test  embedding/obs/test/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/mean/

# eval ds separation Discharge Pretrain model, sectioned by Semantic, section pooling by mean, patient pooling mean and pca
python evaluation_tasks/seperate_datasets.py --obs_train  embedding/obs/train/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/pca/ --obs_test embedding/obs/test/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/pca/ --smokers_train embedding/smokers/train/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/pca/ --smokers_test embedding/smokers/test/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/pca
python evaluation_tasks/seperate_datasets.py --obs_train  embedding/obs/train/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/mean/ --obs_test embedding/obs/test/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/mean/ --smokers_train embedding/smokers/train/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/mean/ --smokers_test embedding/smokers/test/ClinicalBERT_Discharge_Summary_BERT/pretrained/patient/mean_sections/mean