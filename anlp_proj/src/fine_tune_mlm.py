import argparse
import os
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments

from config import FINETUNED_MODELS
from load_data import get_dummy_dataset_by_sections, get_mlm_dataset
from load_models import load_model


def get_process_func(tokenizer, max_tokens):
    def preprocess(record):
        return tokenizer(record['text'], truncation=True, max_length=max_tokens)

    return preprocess

def run_fine_tune_training(model_name, dataset, section_split_type, debug=False):
    tokenizer, model = load_model(model_name, finetuned=False, model_class=AutoModelForMaskedLM)
    output_dir = os.path.join(FINETUNED_MODELS, section_split_type, model.name_or_path.split("/")[-1])
    # todo: read from paper how to treat arguments.
    max_epochs =  3 if debug else 10
    # batch_size = 8 if debug else 8
    training_args = TrainingArguments(output_dir=output_dir,
                                      overwrite_output_dir=True,
                                      evaluation_strategy="epoch",
                                      num_train_epochs=max_epochs,
                                      per_device_train_batch_size=16,
                                      seed=42,
                                      save_strategy='no',  # save last (best)
                                      )

    # Create the DataCollatorForLanguageModeling instance
    data_collator = DataCollatorForLanguageModeling(mlm=True,
                                                    mlm_probability=0.15,
                                                    tokenizer=tokenizer)
    max_tokens = tokenizer.model_max_length
    process_func = get_process_func(tokenizer, max_tokens)
    tokenized_dataset = dataset.map(process_func, batched=True)

    # Split the dataset into train and test sets
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True)
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=tokenized_dataset['train'],
                      eval_dataset=tokenized_dataset['test'],
                      )

    print("Shape of Dataset:\n", tokenized_dataset.shape)
    # Start MLM training
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    # dataset = get_dummy_dataset_by_sections()
    # run_fine_tune_training("dmis-lab/biobert-base-cased-v1.1", dataset, debug=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model name")
    parser.add_argument("--section_split", help="semantic / maximal")
    parser.add_argument("--num_samples", type=int, help="", default=None)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--dummy", type=bool, default=False)

    args = parser.parse_args()
    dataset = get_dummy_dataset_by_sections() if args.dummy else get_mlm_dataset(args.section_split, args.num_samples)
    run_fine_tune_training(args.model, dataset, args.section_split, debug=args.debug)
