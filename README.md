# Opinion Generation using Abstractive Text Summarization

Prerequisites
To run the project make sure you have python 3.7 installed.
Install the below mwntioned libraries:
1) Tensorflow 1.13.1
2) Sklearn
3) tqdm
4) NLTK
5) Keras
6) Pandas
7) gensim
8) numpy

Also download the GoogleNews-vectors-negative300.bin file for word2vec model we have used. Use this file in the multiple scripts that require this file.

Getting started:
Download the dataset given in the data folder. The initial data sets are the yelp_academic_dataset_business and the yelp_academic_dataset_review. 

mult_prep.py
The preprocessing starts on these files. In the mult_prep.py files, give the paths of the above mentioned files (line 23 and 29) and run the process. This file may take 3-4 hours to run and generates a new file which contains all the reviews in a single file - combined_data.txt

write_to_individual_files.py
Use the combined_data.txt file generated in the above step to do the next set of preprocessing. Give the path of this file in the write_to_individual_files.py (line 14) file and execute the script. This script may take upto 6 hours on a normal laptop with 16GB ram 6 GB GPU and a core i7 intel processor. After this scripts run, close to 60,000 files will be generated where each file contains the reviews for a particular restaurant

cleaned_features.py
This file is used to clean the data of every file and generate the cleaned features. Provide the path of the individual review files generated in this file. This generates the features needed in the further steps.

w2v_features.py
This file is used to train the word embedding model using our corpus. This file runs part of the script file birectional_lstm.py

unsupervised_summaries.py
This file generates summaries in an unsupervised way. Choose any file you want to generate a summary of from the handwritten summaries folder and give the path (line 118). This gives an output of short summaries out of which you can pick the highest scored ones.


bidirectional_lstm.py
Is a self sustaining script till all the underlying script are in the working directory.
