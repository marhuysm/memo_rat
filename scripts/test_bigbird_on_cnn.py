# !pip3 install datasets
# !pip3 install rouge_score
# !pip3 install transformers
# !pip3 install sentencepiece

from transformers import PegasusTokenizer, BigBirdPegasusForConditionalGeneration, BigBirdPegasusConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline, SummarizationPipeline, DataCollatorWithPadding

from datasets import load_dataset, load_metric

import torch

model_name = 'google/bigbird-pegasus-large-bigpatent'

loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, attention_type="original_full")

summarizer = SummarizationPipeline(model=loaded_model, tokenizer=loaded_tokenizer)

DATASET_NAME = "bigpatent"
DEVICE = "cuda"
CACHE_DIR = DATASET_NAME
MODEL_ID = f"google/bigbird-pegasus-large-{DATASET_NAME}"

training_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:10%]")
test_dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:10%]")
validation_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:10%]")

# Test : 
#   Afficher la structure des données
#   Récupérer un des textes du corpus, son résumé, et générer
# un nouveau résumé sans pré-entraînement préalable 

print(test_dataset.features)
article = test_dataset["article"][6]
print("================ Article ================ \n")
article = test_dataset["article"][6]
print(article)
print("================ Ref. Summary ================ \n")
refsum = test_dataset["highlights"][6]
print(refsum)
inputs = loaded_tokenizer(article, return_tensors='pt')
summary_ids = loaded_model.generate(**inputs)
print("================ Generated Summary ================ \n")
print(loaded_tokenizer.batch_decode(summary_ids))

# équivalent via pipeline

from transformers import pipeline
summarizer = pipeline("summarization", model=loaded_model, tokenizer=loaded_tokenizer)
result = summarizer(article)
print(result)

# Étape avant l'entraînement : pré-processing du corpus de données d'entraînement, 
# de test et de validation (traiter le dataset avec le tokenizer)

def tokenize_function(example):
    return loaded_tokenizer(example["article"], example["highlights"], truncation=True)

tokenized_training_dataset = training_dataset.map(tokenize_function, batched=True, batch_size=16)

print(tokenized_training_dataset)

tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=16)

print(tokenized_test_dataset)

tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, batch_size=16)

print(tokenized_validation_dataset)

data_collator = DataCollatorWithPadding(tokenizer=loaded_tokenizer)

#Alternative au code précédent, 
# cf https://colab.research.google.com/github/vasudevgupta7/bigbird/blob/main/notebooks/bigbird_pegasus_evaluation.ipynb

DATASET_NAME = "bigpatent"
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
CACHE_DIR = DATASET_NAME
MODEL_ID = f"google/bigbird-pegasus-large-{DATASET_NAME}"

def generate_answer(batch):
  inputs_dict = loaded_tokenizer(batch["article"], padding="max_length", max_length=4096, return_tensors="pt", truncation=True)
  inputs_dict = {k: inputs_dict[k].to(DEVICE) for k in inputs_dict}
  predicted_abstract_ids = loaded_model.generate(**inputs_dict, max_length=256, num_beams=5, length_penalty=0.8)
  batch["predicted_abstract"] = loaded_tokenizer.decode(predicted_abstract_ids[0], skip_special_tokens=True)
  print(batch["predicted_abstract"])
  return batch

dataset_small = test_dataset.select(range(2))
result_small = dataset_small.map(generate_answer)

rouge = load_metric("rouge")

rouge.compute(predictions=result_small["predicted_abstract"], references=result_small["highlights"])

# Test sur la phrase testée en début de code : 

dataset_small = test_dataset.select([6])
result_small = dataset_small.map(generate_answer)

rouge = load_metric("rouge")

rouge.compute(predictions=result_small["predicted_abstract"], references=result_small["highlights"])

# # Évaluation sur 600 textes :

# test_dataset = test_dataset.select(range(600))

# result = test_dataset.map(generate_answer)

# rouge.compute(predictions=result["predicted_abstract"], references=result["highlights"])

## Objectif après l'évaluation : entraîner, puis sauvegarder le nouveau modèle, et le réévaluer

# from transformers import Trainer, TrainingArguments

# rouge = load_metric("rouge")

# training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

# def tokenize_function(example):
#     return loaded_tokenizer(example["article"], example["highlights"], truncation=True)

# tokenized_training_dataset = training_dataset.map(tokenize_function, batched=True, batch_size=16)

# print(tokenized_training_dataset)

# tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=16)

# print(tokenized_test_dataset)

# tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, batch_size=16)

# print(tokenized_validation_dataset)

# data_collator = DataCollatorWithPadding(tokenizer=loaded_tokenizer)

# import numpy as np
# from datasets import load_metric

# predictions = trainer.predict(tokenized_validation_dataset)
# print(predictions.predictions.shape, predictions.label_ids.shape)


# def compute_metrics(eval_preds):
#     metric = load_metric("rouge")
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


# trainer = Trainer(
#     loaded_model,
#     training_args,
#     train_dataset=tokenized_training_dataset,
#     eval_dataset=tokenized_validation_dataset,
#     data_collator=data_collator,
#     tokenizer=loaded_tokenizer,
#     compute_metrics=compute_metrics,
# )

# trainer.train()