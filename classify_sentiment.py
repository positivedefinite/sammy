import data_helpers_neutrals
import pandas as pd
import numpy as np
#from keras.preprocessing import sequence
#from gensim.models import word2vec
#from os.path import join, exists, split
#import os
#import h5py
from keras.models import load_model

"""
This module gives a prediction (betwen 0 and 1) for a set of input sentences and calculates accuracy on whole set.
INPUT: a keras model, a vocabulary dictionary and a file with tweets to test
OUTPUT: accuracy
"""

# import vocabulary
input_data = np.load('C:/AI/eh_CNN_twitter/data/semeval/input_data.npy')
#x_train = input_data[0]
#y_train = input_data[1]
#x_test = input_data[2]
#y_test = input_data[3]
vocabulary_inv = input_data[4]
#neutral_tweets = input_data[5]
vocabulary = {y:x for x,y in vocabulary_inv.items()}


# import CNN model
model = load_model('C:/AI/eh_CNN_twitter/models/mixedTest.h5')


def label_tweets(comp_name):
    """ this function uses the trained model and labels new tweets."""
    # import company tweets
    fpath = 'C:/AI/eh_CNN_twitter/data/comp_tweets/'
    comp_tweets = pd.read_csv(fpath+comp_name+'.tsv', sep='\t', encoding = 'utf-8',  header=None, error_bad_lines=False)#, nrows = 10) 
    comp_tweets_original = comp_tweets.copy(deep= True)
    comp_tweets = comp_tweets.iloc[:,5]
    comp_tweets = [str(s).strip() for s in comp_tweets]
    comp_tweets = [data_helpers_neutrals.clean_str(sent) for sent in comp_tweets]
    comp_tweets = [s.split(" ") for s in comp_tweets]
    comp_tweets = pad_tweets(comp_tweets)
    print(comp_name, ': cleaned and padded')
    comp_tweets = [[vocabulary[word] for word in tweet] for tweet in comp_tweets]
    y = model.predict(np.array(comp_tweets), batch_size=1) 
    print(comp_name, ':prediction done!')
    comp_tweets_labeled = pd.DataFrame(y)
    comp_tweets_labeled = comp_tweets_labeled.rename(columns={0:"prediction"})
    comp_tweets_labeled['tweet'] = comp_tweets_original.iloc[:,5]
    comp_tweets_labeled['Date'] = comp_tweets_original.iloc[:,2]
#    comp_tweets_labeled['prediction'] = np.round(comp_tweets_labeled['prediction'], 2)
    comp_tweets_labeled.to_csv(fpath+comp_name+'_labeled'+'.tsv', sep='\t', encoding='utf-8')
    return comp_tweets_labeled




# padding tweets
def pad_tweets(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = 40
    print('Padding length determined as ' +str(sequence_length))
    
    
    padded_sentences = [] # PREALLOCATE FOR SPEED
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) > sequence_length:
            new_sentence = sentence[0:sequence_length]
        else:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences



comp_names = ['ing_tweets', 'monsanto_tweets', 'mylan_tweets', 'samsung_tweets', 'telia_tweets', 'volkswagen_tweets']
for comp_name in comp_names:
    comp_tweets_labeled = label_tweets(comp_name)
    







