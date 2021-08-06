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

raw_dataset_en = load_dataset("cnn_dailymail", "3.0.0")
dataset_en = raw_dataset_en['validation']
model_name_en = "flax-community/t5-base-cnn-dm"
loaded_tokenizer_en = AutoTokenizer.from_pretrained(model_name_en)
loaded_model_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_en)
summarizer_en = pipeline("summarization", model=loaded_model_en, tokenizer=loaded_tokenizer_en)

result_en = []
evaluation_result = []

metric = load_metric("rouge")

for i in range(0, dataset_en.dataset_size):
    article = dataset_en[i]
    article_copy = copy.deepcopy(article)
    ref_summ = dataset_en['highlights'][i]

    text = dataset_en['article'][i]

    if len(text) > 0 : 
        gen_summ = summarizer_en(text, truncation= True)
        result_en = gen_summ[0]['summary_text']

        rouge = Rouge()
        rouge_score = rouge.get_scores(result_en, ref_summ)
        
        article_copy['gen_summary'] = result_en
        article_copy['rouge_score'] = rouge_score

        evaluation_result.append(article_copy)

    print("Progress: ", i, '/', len(dataset_en))

    if (i == 600):
      break

with open('en_t5_cnn_results.json', 'w', encoding='utf8') as out_file:
    out = json.dumps(evaluation_result, ensure_ascii=False, indent=2)
    out_file.write(out)