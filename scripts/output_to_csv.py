import json
import csv
 
with open('sample_en_t5_cnn_results.json') as json_file: # ou 'final_output_en.json', etc
    data = json.load(json_file)
 
data_file = open('csv_copy_sample_en_t5_cnn_results.csv', 'w') # ou 'sample_en.csv, ...
 
csv_writer = csv.writer(data_file)
 
count = 0
sample = 0

for article in data:
    if count == 0:
 
        header = article.keys()
        csv_writer.writerow(header)
        count += 1
 
    # Writing data of CSV file
    if sample < 50 :        # 50 = le nombre de documents que l'on veut dans le csv
        csv_writer.writerow(article.values())
        sample += 1
 
data_file.close()