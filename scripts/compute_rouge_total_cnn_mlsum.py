import json

from rouge import Rouge

f = open('en_t5_cnn_results.json' ) # ou 'fr_t5_mlsum_results.json'

data = json.load(f)

ref_summaries = []
summaries = []

for article in data:
    ref_summaries.append(article['highlights']) # pour mlsum : summary, pour cnn : highlights
    summaries.append(article['gen_summary'])

print(len(ref_summaries))

rouge = Rouge()
rouge_score = rouge.get_scores(summaries, ref_summaries, avg=True)

print("ROUGE SCORE = ", rouge_score)

f.close()