# Scripts utilisés pour l'étude de cas du mémoire "Analyse de la performance des modèles de résumé automatique par abstraction en langue française"

Ce repository contient les scripts qui ont été utilisés dans le cadre de mon mémoire de Master en Sciences et Technologies de l'Information et de la Communication à l'ULB (2020-2021).

## Présentation de l'étude de cas

L'objectif de mon étude de cas est d'analyser la performance d'un modèle de langue neuronal entraîné pour le résumé automatique de textes par abstraction, plus particulièrement en ce qui concerne le français.
Pour ce faire, j'ai choisi de comparer la performance du modèle T5 en anglais et en français sur base des corpus Wikilingua (français et anglais), CNN-Dailymail (anglais) et MLSUM (français) en passant par les outils mis à disposition via la librairie HuggingFace.

### Note

J'ai d'abord voulu analyser les performances du modèle BigBird-Pegasus sur un corpus de langue française.
Mon objectif premier était d'analyser les performances sans pré-entraînement sur le corpus CNN-DailyMail, pour ensuite procéder à un fine-tuning supplémentaire sur ce même corpus et réévaluer les résultats, et faire de même en français sur base du corpus MLSUM.
Mais j'ai été bloquée à cause de limitations techniques : même en utilisant la plateforme Google Colaboratory, une évaluation sur 600 textes était impossible : en 5h14, seuls 115 textes avaient été traités, ce qui rendait le reste de l'expérience impossible à réaliser.
J'ai toutefois inclus ici un script résumant ce premier essai.

## Déroulement de l'étude de cas

Mon étude de cas s'est déroulée en plusieurs étapes : 

Un échantillon de 600 documents par corpus a été créé.
Chaque texte de l'échantillon a été résumé en utilisant le modèle T5 correspondant à sa langue ([plguillou/t5-base-fr-sum-cnndm](https://huggingface.co/plguillou/t5-base-fr-sum-cnndm) pour les corpus en français et [flax-community/t5-base-cnn-dm](https://huggingface.co/flax-community/t5-base-cnn-dm) pour les corpus en anglais), et les scores ROUGE-1, ROUGE-2 et ROUGE-L ont été calculés sur base des résumés de référence disponibles dans les corpus.
Les résultats pour chaque corpus sont enregistrés dans un fichier json contenant les informations disponibles dans le corpus, les résumés générés et les scores ROUGE pour chaque document.
Les fichiers json sont disponibles dans le dossier `data`.

J'ai également calculé les scores ROUGE moyens pour chaque échantillon de corpus.

Enfin, j'ai procédé à une extraction sur base des fichiers json vers un document csv afin de pouvoir facilement accéder à un plus petit échantillon que j'ai utilisé pour l'analyse qualitative des résultats obtenus.
L'analyse qualitative est basée sur 240 résultats en tout : j'ai extrait 70 textes (25 documents) pour les corpus Wikilingua français et anglais, 50 textes pour le corpus MLSUM, et 50 textes pour le corpus CNN.

## Environnement

J'ai fait tourner tous les scripts python dans un environnement virtuel en utilisant les commandes ci-dessous : 

```
brew install python3 # Spécifique à macos
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
```

Il est nécessaire d'installer les packages suivants pour pouvoir utiliser les scripts : 

```
pip3 install transformers
pip3 install datasets
pip3 install rouge_score
pip3 install rouge
pip3 install sentencepiece
pip3 install torch
```

## Présentation des scripts

 `test_bigbird_on_cnn.py` : ma première tentative infructueuse telle que décrite dans la note ci-dessus.

 `en-t5_wikilingua.py` : script de création de l'échantillon d'analyse sur le corpus wikilingua anglais, y compris les scores ROUGE.
  
 `fr-t5_wikilingua.py` : script de création de l'échantillon d'analyse sur le corpus wikilingua français, y compris les scores ROUGE.
  
 `en_t5_cnn` : script de création de l'échantillon d'analyse sur le corpus CNN-Dailymail, y compris les scores ROUGE.
 
 `fr-t5-mlsum.py` : script de création de l'échantillon d'analyse sur le corpus MLSUM français, y compris les scores ROUGE.
 
 `compute_rouge_total_wikilingua.py` : calcul du score ROUGE moyen sur base des fichiers json créés par les scripts précédents (valable pour les deux corpus wikilingua).
 
 `compute_rouge_total_cnn_mlsum.py` : calcul du score ROUGE moyen sur base des fichiers json créés par les scripts précédents (valable pour les corpus CNN-Dailymail et MLSUM).
 
 `output_to_csv.py` : extraction d'un échantillon pour l'analyse qualitative à partir des fichiers json vers un fichier csv.