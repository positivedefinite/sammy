# Part 1: Extract Transform and Load
# Assumptions: tweets are stored in a tsv file

from __future__ import absolute_import, division, print_function
#import pandas as pd
import ETL
import time
import numpy as np
from w2v import train_word2vec
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, ZeroPadding1D
from keras.layers.merge import Concatenate
#from keras.datasets import imdb
#from keras.preprocessing import sequence
from keras import callbacks
#from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

settings = {'file_name' : time.strftime("%Y%m%d-%H%M%S-")+' Default Experiment', 
            'default_run' : True, 
            'load_data' : True,
            'save_stats' : True, 
            'save_model' : True,
	         'dataset_name': ['semeval_en']
            }

print('Current Settings are as follows : \n' + '\tFile Name = ' + str(settings['file_name']) + '\n\tLoad data from .npy file = ' + str(settings['load_data']))
print('\tSave Stats file = ' + str(settings['save_stats']) + '\n\tSave Model file = ' + str(settings['save_model']) + '\n\tDataset Name = ' + str(settings['dataset_name'][0]))
do_you_want_a_default_run = input("Do you want to change these settings? (Y/N) ----> ")
while do_you_want_a_default_run not in ['Y','N','y','n']:
    do_you_want_a_default_run = input("Sorry that input was invalid, please enter either Y or N. Do you want to change any of these settings? ")
if do_you_want_a_default_run in ['Y','y']:
    settings['default_run'] = False
    # Load data
    load_data = input("Do you want to load data from .npy file? (Y/N) ----> ")
    while load_data not in ['Y','N','y','n']:
    	load_data = input("Sorry that input was invalid, please enter either Y or N. Do you want to save the STATS file ----> ")
    if load_data in ['N','n']:
        settings['load_data']=False
    # Save statistics
    save_stats = input("Do you want to save the STATS file from this experiment (Y/N) ----> ")
    while save_stats not in ['Y','N','y','n']:
    	save_stats = input("Sorry that input was invalid, please enter either Y or N. Do you want to save the STATS file ----> ")
    if save_stats in ['Y','y']:
        settings['save_stats']=True
    else:
        settings['save_stats']=False
    # Save model
    save_model = input("Do you want to save the MODEL for this experiment (Y/N) ----> ")
    while save_model not in ['Y','N','y','n']:
    	save_model = input("Sorry that input was invalid, please enter either Y or N. Do you want to save the MODEL file ----> ")
    if save_model in ['Y','y']:
        settings['save_model']=True
    else:
        settings['save_model']=False
    change_name = input("Do you want to change the name for this experiment (Y/N) ----> ")
    while change_name not in ['Y','N','y','n']:
    	change_name = input("Sorry that input was invalid, please enter either Y or N. Do you want to change the name for this experiment ----> ")
    if change_name in ['Y','y']:
        experiment_name = input("Enter a short yet descriptive name for this experiment:\n----> ")
        timestr = time.strftime("%Y%m%d-%H%M%S ")
        settings['file_name'] = timestr + experiment_name
    print('New Settings are as follows : \n' + '\tFile Name = ' + str(settings['file_name']) + '\n\tLoad data from .npy file = ' + str(settings['load_data']))
    print('\tSave Stats file = ' + str(settings['save_stats']) + '\n\tSave Model file = ' + str(settings['save_model']) + '\n\tDataset Name = ' + str(settings['dataset_name'][0])+'\n')
#loads both annotated and neutral tweets
if settings['load_data']==False:
    x_train, y_train, x_test, y_test, vocabulary_inv, neutral_tweets = ETL.main(settings['dataset_name'], 1, 0.8)
    np.save('data/'+str(settings['dataset_name'][0])+'/'+'input_data.npy',[x_train, y_train, x_test, y_test, vocabulary_inv, neutral_tweets])
else:
    print('Loading data...')
    input_data = np.load('data/'+str(settings['dataset_name'][0])+'/input_data.npy')
    x_train = input_data[0]
    y_train = input_data[1]
    x_test = input_data[2]
    y_test = input_data[3]
    vocabulary_inv = input_data[4]
    neutral_tweets = input_data[5]
    print('data has been loaded!')


### Part 2A: Hyperparameters
"""
Original convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf
Our team has taken this code from https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras and adjusted the hyperparameters
"""

np.random.seed(0)

model_type = "CNN-non-static"  

# Data source
data_source = "local_dir"

# Model Hyperparameters
filter_sizes = (3, 8)
num_filters = 20 #10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 2

# Prepossessing parameters
sequence_length = 140 #400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
embedding_dim = 50
min_word_count = 10
context = 5
# ---------------------- Parameters end -----------------------
print("Completed.")

if settings['save_stats'] == True:
	print("Generating stat file...")
	statfile = open("./models/stats/" + settings['file_name'] + "-stats.txt",'w')
	statfile.write("EXPERIMENT: " + settings['file_name'] + "\nModel type= " + str(model_type) + "\nNumber of epochs = " + str(num_epochs) )
	statfile.write( "\nBatch Size = " + str(batch_size) + "\nNumber of embeddings dimensions = " + str(embedding_dim) + "\nFilter sizes: " + str(filter_sizes))
	statfile.write("\nNumber of filters = " + str(num_filters) + "\nDropout probabilities: " + str(dropout_prob) + "\nNumber of hidden dimensions = " + str(hidden_dims))
	statfile.write("\nSequence Length = " + str(sequence_length) + "\nMaximum num words: " + str(max_words) + "\nMinimum word count = " + str(min_word_count) + "\nContext = " + str(context))
	statfile.write("\n\nAccuracy results for Epochs run:\n")
	statfile.close()
	print("Stat file created !!")

    
### Part 2B: Network definition & word2vec training
### make sure to delete existing word2vec model if you want to udate it
if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]

print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    print('Initiating word2vec.')
    embedding_weights = train_word2vec(np.vstack((x_train, x_test, neutral_tweets)), settings['dataset_name'], vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    print('Word2vec done.')
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
        print("x_train static shape:", x_train.shape)
        print("x_test static shape:", x_test.shape)

elif model_type == "CNN-rand":
    embedding_weights = None
else:
    raise ValueError("Unknown model type")


# Build model
if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,)

model_input = Input(shape=input_shape)

# Static model does not have embedding layer
if model_type == "CNN-static":
    z = model_input
else:
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

z = Dropout(dropout_prob[0])(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = ZeroPadding1D(padding = int((sz-1)/2))(z)
    conv = Convolution1D(filters=num_filters,
                            kernel_size=sz,
                            padding="valid",
                            activation="relu",
                            strides=1)(conv)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)

z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)

model_output = Dense(1, activation="sigmoid")(z)
model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# Initialize weights with word2vec
if model_type == "CNN-non-static":
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])


# Part 3: Training the model
# one epoch performs better on new data
# on 16th of Oct, training with 350k annon and 130k neutral took about 260sec

if settings['save_stats'] == True:
	csv_logger = callbacks.CSVLogger("./models/stats/" + settings['file_name'] + "-stats.txt", separator='\t', append=True)
	model.fit(x_train, y_train, batch_size=5, epochs=num_epochs, validation_data = (x_test, y_test), callbacks = [csv_logger], verbose=1)
else:
	model.fit(x_train, y_train, batch_size=5, epochs=num_epochs, validation_data = (x_test, y_test), verbose=1)

    
if settings['save_model'] == True:
	#model.save("./models/" + settings['file_name'] + ".h5" )
	##Serialize model to json
	model_json = model.to_json()
	with open('./models/'+settings['file_name']+'_json'+'.json','w') as json_file:
		json_file.write(model_json)
	##Serialize model weights to HDF5
	model.save_weights('./models/'+settings['file_name']+'_weights'+'.h5')
	print('Model saved!!')

val_accu = model.evaluate(x_test,y_test,verbose = 0)
print("Accuracy = " + str(val_accu[1]))

y_confuse = model.predict(x_test,batch_size = batch_size,verbose = 1)
for i in range(len(y_confuse)):
    if y_confuse[i]>0.5:
        y_confuse[i] = 1
    else:
        y_confuse[i] = 0

confuse_matrix = metrics.confusion_matrix(y_test, y_confuse)
print("\nConfusion Matrix:")
print(str(confuse_matrix))

experiments_log = open('experiments.tsv','a',encoding = 'utf-8')
experiments_log.write(str(settings['file_name'])+str(confuse_matrix)+"\n")
experiments_log.close()

if settings['save_stats'] == True:
	statfile = open("./models/stats/" + settings['file_name'] + "-stats.txt",'a+')
	statfile.write("\n\n Confusion matrix: \n" + str(confuse_matrix) + "\n\n")
	statfile.write("Accuracy = " + str(val_accu[1]))
	statfile.close()
    
    
    
