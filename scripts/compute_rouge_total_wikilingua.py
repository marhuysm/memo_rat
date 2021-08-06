import json

from rouge import Rouge

f = open('final_output_fr.json' ) # ou 'final_output_en.json'

data = json.load(f)

ref_summaries = []
summaries = []

for article in data:
    ref_summaries = ref_summaries + article['ref_summary']
    summaries = summaries + article['summary']

print(len(ref_summaries))

rouge = Rouge()
rouge_score = rouge.get_scores(summaries, ref_summaries, avg=True)

print("ROUGE SCORE = ", rouge_score)

f.close()