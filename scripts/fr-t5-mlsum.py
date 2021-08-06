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

raw_dataset_fr = load_dataset("mlsum", "fr")
dataset_fr = raw_dataset_fr['validation']
model_name_fr = "plguillou/t5-base-fr-sum-cnndm"
loaded_tokenizer_fr = AutoTokenizer.from_pretrained(model_name_fr)
loaded_model_fr = AutoModelForSeq2SeqLM.from_pretrained(model_name_fr)
summarizer_fr = pipeline("summarization", model=loaded_model_fr, tokenizer=loaded_tokenizer_fr)

result_fr = []
evaluation_result = []

metric = load_metric("rouge")

for i in range(0, dataset_fr.dataset_size):
    article = dataset_fr[i]
    article_copy = copy.deepcopy(article)
    ref_summ = dataset_fr['summary'][i]

    text = dataset_fr['text'][i]

    if len(text) > 0 : 
        gen_summ = summarizer_fr(text, truncation= True)
        result_fr = gen_summ[0]['summary_text']

        rouge = Rouge()
        rouge_score = rouge.get_scores(result_fr, ref_summ)
    
        article_copy['gen_summary'] = result_fr
        article_copy['rouge_score'] = rouge_score

        evaluation_result.append(article_copy)

    print("Progress: ", i, '/', len(dataset_fr))

    if (i == 600):
      break

with open('fr_t5_mlsum_results.json', 'w', encoding='utf8') as out_file:
    out = json.dumps(evaluation_result, ensure_ascii=False, indent=2)
    out_file.write(out)