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

raw_dataset_en = load_dataset("wiki_lingua", "english")
dataset_en = raw_dataset_en['train']
model_name_en = "flax-community/t5-base-cnn-dm"
loaded_tokenizer_en = AutoTokenizer.from_pretrained(model_name_en)
loaded_model_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_en)
summarizer_en = pipeline("summarization", model=loaded_model_en, tokenizer=loaded_tokenizer_en)

result_en = []
evaluation_result_en = []

metric = load_metric("rouge")

# Si l'on veut passer un pourcentage à la place du 600 à la ligne
# for article in dataset_fr['article'][:600] :
percentage = .1
num_process = math.floor(len(dataset_en) / 100 * percentage)

num_articles = 0
for article in dataset_en['article'][:600]:
    article_copy = copy.deepcopy(article)
    num_articles += 1
    ref_summ = article.get('summary')

    text = article.get('document')
    if len(text) > 0 : 
        result_en = summarizer_en(text, truncation= True)
        summaries = [] 
        for summary in result_en:
            summaries.append(summary['summary_text'])

        rouge = Rouge()
        rouge_score = rouge.get_scores(summaries, ref_summ, avg= True)

        article_copy['ref_summary'] = article_copy['summary']
        article_copy['summary'] = summaries
        article_copy['rouge_score'] = rouge_score

        evaluation_result_en.append(article_copy)

    print("Progress: ", num_articles, '/', len(dataset_en['article']))

with open('final_output_en.json', 'w', encoding='utf8') as out_file:
    out = json.dumps(evaluation_result_en, ensure_ascii=False, indent=2)
    out_file.write(out)
