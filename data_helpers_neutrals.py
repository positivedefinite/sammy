import re
import itertools
from collections import Counter
import numpy as np
import pandas as pd

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""
def load_data(howMuchToLoad, datasets):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    dataset_name=datasets[0]
    # Load and preprocess data
    print('Processing dataset '+str(dataset_name))
    neutral_tweets = load_neutral_data(dataset_name)
    print('Step 1/5: Neutral loading done')
    all_annotated_tweets, all_labels = load_data_and_labels(dataset_name)
    reqd_median = int(len(all_annotated_tweets)*howMuchToLoad*0.5)
    annotated_tweets = all_annotated_tweets[:reqd_median]
    labels = all_labels[:reqd_median]
    annotated_tweets = np.append(annotated_tweets,all_annotated_tweets[-reqd_median:])
    labels= np.concatenate([labels,all_labels[-reqd_median:]],0)
    newMax = len(annotated_tweets)
    print('Taking only '+str(howMuchToLoad*100)+'% of available labelled data. Number of tweets is '+str(newMax))
    
    print('Step 2/5: Annotated loading done')
    #sentences_padded = pad_sentences(sentences)
    [annotated_padded, neutral_padded] = pad_tweets(annotated_tweets, neutral_tweets)
    print('Step 3/5: Padding done with '+str(len(annotated_padded))+' annotated tweets and '+str(len(neutral_padded))+' neutral tweets.')
    vocabulary, vocabulary_inv = build_vocab(annotated_padded + neutral_padded)
    print('Step 4/5: Vocabulary building done.')
    x, y = build_input_data(annotated_padded, labels, vocabulary)
    z = build_neutral_data(neutral_padded, vocabulary)
    print('Step 5/5: Data loaded as numpy array.')
    return [x, y, vocabulary, vocabulary_inv, z]


def load_data_and_labels(dataset_name):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    for i in [1]:
        # Load data from files
        positive_examples = list(open('data/'+str(dataset_name)+'/'+str(dataset_name)+'.pos',encoding="utf-8").readlines())
#        positive_examples = positive_examples[0:1000]
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open('data/'+str(dataset_name)+'/'+str(dataset_name)+'.neg',encoding="utf-8").readlines())
#        negative_examples = negative_examples[0:1000]
        negative_examples = [s.strip() for s in negative_examples]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [clean_str(sent) for sent in x_text]
        x_text = [s.split(" ") for s in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_neutral_data(dataset_name):
#    neutral_examples = list(open('data/'+str(dataset_name)+'/'+str(dataset_name)+'.neu', encoding="utf-8").readlines())
#    neutral_examples = pd.read_csv('data/'+str(dataset_name)+'/'+'sam_data.tsv', sep='\t', encoding = 'utf-8',  header=None, error_bad_lines=False) 
#    neutral_examples = pd.read_csv('data/'+str(dataset_name)+'/'+'eng_sam.tsv', sep='\t', encoding = 'utf-8',  header=None, error_bad_lines=False, nrows=1000)
    neutral_examples = pd.read_csv('data/semeval_combo/'+'semeval_combo.neu', sep='\t', encoding = 'utf-8',  header=None, error_bad_lines=False)#, nrows=10000)
    shuffle_indices = np.random.permutation(np.arange(len(neutral_examples)))
    neutral_examples = neutral_examples.iloc[shuffle_indices]
#    neutral_sample = int(len(neutral_examples)*0.10)
#    neutral_examples = neutral_examples[0:neutral_sample]
#    neutral_examples_temp = neutral_examples[[2,5]][neutral_examples[11] == "EN"]
#    neutral_examples = list(neutral_examples.iloc[:,5])
    neutral_examples = [str(s).strip() for s in neutral_examples]
    neutral_tweets = [clean_str(sent) for sent in neutral_examples]
    neutral_tweets = [s.split(" ") for s in neutral_tweets]
    return neutral_tweets


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "_url_", string)
    string = re.sub(r'@([^\s:]+)',"_usernaem_", string)
    emot = dict.fromkeys([':-)',':)','(:','(-:',':-))'],'_smileyFace_')
    emot.update(dict.fromkeys([':-D',':D','=D','XD','xD'],'_laughing_'))
    emot.update(dict.fromkeys([':-(',':(','):',')-:'],'_sadFace_'))
    emot.update(dict.fromkeys([':-P',':P',':-p',':p'],'_tongueOut_'))
    emot.update(dict.fromkeys([':-O',':O',':-o',':o'],'_surprised_'))
    emot.update(dict.fromkeys([':-$',':$','$:','$-:'],'_embarrassedFace_'))
    emot.update(dict.fromkeys([':-/',':/',':\\',':-\\'],'_skeptical_'))
    emot.update(dict.fromkeys([':-|',':|'],'_straightFace_'))
    emot.update(dict.fromkeys([':\'-(',':\'('],'_crying_'))
    emot.update(dict.fromkeys([':\'-)',':\')'],'_happyTears_'))
    emot.update(dict.fromkeys([';-)',';)','*-)','*)',';D'],'_wink_'))
    for key in emot:
        string = string.replace(key,emot[key])
        
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = [] # PREALLOCATE FOR SPEED
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def pad_tweets(annotated, neutral, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    #sequence_length = max([max(len(x) for x in annotated), max(len(x) for x in neutral)])
    sequence_length = 40
    print('Padding length determined as ' +str(sequence_length))
    
    def pad(sentences, sequence_length):
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
    
    annotated_padded = pad(annotated,sequence_length)
    neutral_padded = pad(neutral,sequence_length)
    return [annotated_padded, neutral_padded]


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def build_neutral_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    z = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return z


# this is not used at all
def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



