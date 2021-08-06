# !pip install transformers 
# !pip install datasets 
# !pip install rouge_score 
# !pip install rouge 
# !pip install torch 
# !pip install sentencepiece

import math
import csv
import copy
import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from datasets import load_dataset
from datasets import load_metric

from rouge import Rouge

raw_dataset_fr = load_dataset("wiki_lingua", "french")
dataset_fr = raw_dataset_fr['train']
model_name_fr = "plguillou/t5-base-fr-sum-cnndm"
loaded_tokenizer_fr = AutoTokenizer.from_pretrained(model_name_fr)
loaded_model_fr = AutoModelForSeq2SeqLM.from_pretrained(model_name_fr)
summarizer_fr = pipeline("summarization", model=loaded_model_fr, tokenizer=loaded_tokenizer_fr)

result_fr = []
evaluation_result = []

metric = load_metric("rouge")

# Si l'on veut passer un pourcentage à la place du 600 à la ligne
# for article in dataset_fr['article'][:600] :
percentage = .1
num_process = math.floor(len(dataset_fr) / 100 * percentage)

num_articles = 0
for article in dataset_fr['article'][:600]:
    article_copy = copy.deepcopy(article)
    num_articles += 1
    ref_summ = article.get('summary')

    text = article.get('document')

    if len(text) > 0 : 
        result_fr = summarizer_fr(text, truncation= True)
        summaries = [] 
        for summary in result_fr:
            summaries.append(summary['summary_text'])

        rouge = Rouge()
        rouge_score = rouge.get_scores(summaries, ref_summ, avg= True)

        article_copy['ref_summary'] = article_copy['summary']
        article_copy['summary'] = summaries
        article_copy['rouge_score'] = rouge_score

        evaluation_result.append(article_copy)

    print("Progress: ", num_articles, '/', len(dataset_fr['article']))

with open('final_output_fr.json', 'w', encoding='utf8') as out_file:
    out = json.dumps(evaluation_result, ensure_ascii=False, indent=2)
    out_file.write(out)