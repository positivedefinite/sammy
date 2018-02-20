# Sammy
<p align="center">
  <img src="http://cubexsys.com/wp-content/uploads/2017/09/Words.png">
</p>

Sammy is a Sentiment Analysis Model for mixed tweets in dutch and english. It has an accuracy of 80% on a balanced set of positive and negative tweets with a symmetric error (similar number of false positives and false negatives). It is developed and maintained by the AI Team @ GroeiFabriek APG in Heerlen. Contributors are: Oskar Person, Praveen Koshy Sam, Bart Driessen, Ebrahim Hazrati and Koen Weterings.

This convolutional neural network is based on Alexander Rakhlin's repository that can be found here: https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras. We have adapted his code to handle twitter data instead of imdb, fine-tuned the metaparametrs and wrote a lot of code around the original network to handle mixed language tweets. In our repository you can also find code for easily predicting sentiment for thousands of tweets with pre-existing models. 

You can find our best model, original semeval twitter data and semeval data translated to dutch, at the dropbox: https://www.dropbox.com/sh/hka08rfh2n7d9bc/AAA7R1kPSzYlG47pLRrxlOKXa?dl=0

We have also created a knowledge sharing presentation explaining the workings of a CNN to our fellow colleagues in data science. You can contact us if you wish to use those in-depth educational materials.

## The Code
There are 4 modules in the main code:

***1.	sammy:*** This has the bulk of the code and calls the relevant modules as required. The hyperparameters for the model are set here and all files generated during the code execution are generated from here.

***2.	ETL:*** This is very minimal module that simply exists to call the data_helpers_neutrals module and to format the output.

***3.	data_helpers_neutrals:*** Here the data is loaded and cleaned. It also builds required data structures such as the vocabulary.

***4.	w2v:*** This model checks if the required word2vec model exists and if not it runs the word2vec model to train the word vectors.

<p align="center">
  <img src="https://github.com/positivedefinite/sammy/blob/master/execution_flow.png">
</p>

## Main Code Modules in depth
### 1.	sammy
This is the main module for execution. There are no functions defined in it. The non-standard python packages it depends on are keras, scikit-learn, numpy and H5PY. It starts by setting parameters in a dictionary ‘settings’. The user is given two options run in default mode or change settings. The default run sets the name of the file as ‘<time> Default Experiment’, it searches for and loads a .npy file in a pre-defined location and does not save the stats nor the model file. In a non-default run the user can choose to train a new .npy file and to save both the stats file and the model file. The user will also have to specify a name for the experiment. 
After this if loading .npy file is disabled, the program will call ETL and load the tweets from the dataset mentioned. After loading the tweets are saved in a .npy file in the folder in ‘data’ that has the same name as the dataset being loaded.
The program then sorts the data into the respective variables and separates the training set and the test set. The hyperparameters for the model are then set and the model type of ‘CNN-non-static’ is selected. This model type means that the word embedding layer is optimized as the training is performed.
The ‘stats’ file is created and then the word2vec model is trained via the w2v model.
After this the layers for the convolutional neural network are defined. It first has the embedding layer followed by a dropout layer, then for each ‘filter_size’ it first does zero_padding, followed by a 1-D convolutional layer, which is then max_pooled and flattened. The outputs for the different filter_sizes are concatenated and dropout is performed again. The output from the dropout is passed through one dense layer with ReLU activation and through one dense layer with Sigmoid activation. The model is then compiled with ‘binary_crossentropy’ loss and the ‘adam’ optimizer. For a setup where there are two ‘filter_size’ defined, the network would like as shown below.

<p align="center">
  <img src="https://github.com/positivedefinite/sammy/blob/master/model_architecture.png">
</p>

The accuracy measurements are logged in a text file during the model training. The model developed is then saved in two parts. The model architecture is saved in a JSON file and the model weights are saved in .h5 file. The confusion matrix is generated and saved in the text file as well.
The model files are saved in the ‘models’ folder and the stats files are stored in the ‘models/stats/’ folder

### 2.	ETL
The ETL module has two functions. The ‘main’ function and the ‘load_data’ function.
The ‘main’ function is referenced in sammy and it takes 3 arguments – the name of the dataset, fraction of dataset to load (0.0 – 1.0) and the fraction of how much of the dataset should be set for training. It then calls the ‘load_data’ function and returns the training data, training set labels, test data, test set labels, the vocabulary as a dictionary with the key as a unique integer with every unique word in the dataset as value and the neutral data (data without labels).
The ‘load_data’ function takes the same arguments as the ‘main’ function. It then calls the eponymous function in data_helpers_neutrals from which it receives the training data, the labels, the vocabulary and the neutral data. It then shuffles the data and splits it into the fraction specified by the third argument in the function. It returns the training data, training set labels, test data, test set labels, the vocabulary as a dictionary with the key as a unique integer with every unique word in the dataset as value and the neutral data (data without labels).

### 3.	data_helpers_neutrals
The data_helpers_neutrals module has 10 functions. Below you will find a description of each function.

***i.	load_data***
This is the main function for this module that is referenced in the ETL module. It takes two arguments – fraction of dataset to load and the name of the dataset. It returns 5 objects – the annotated data, the labels for the annotated data, a dictionary with a unique word as key and a unique integer as the value, the list of unique words in the data and the neutral data. 
The function loads data from the folder with the name set as the argument by calling the load_data_and_labels function. It then truncates the data based on the value set as the first argument to the function. The pad_tweets function is then called to pad each entry to the same length. The build_vocab function is called to generate the vocabulary from the data. Next up the build_input_data function is called to separate the tweets and the labels. The build_neutral_data function is then called to save the neutral data.

***ii.	load_data_and_labels***
This function takes one argument – the name of the dataset to be loaded and returns the text from the files and the labels.
It assumes there are two different files for each dataset – a positive data file and a negative data file. It will open both files and read the data and concatenates them into one variable. clean_str is then called to clean the data and to make it match an expected format. The labels are also concatenated to one variable.

***iii.	load_neutral_data***
This function takes one argument – the name of the dataset to be loaded and returns the neutral data from the respective file. The location for the neutral data has to be manually specified for this. It shuffles the neutral data and then runs clean_str to make it adhere to the specified format. It returns the cleaned neutral data after.

***iv.	clean_str***
This function takes a string and returns the cleaned string. For cleaning it using regular expressions via the python package ‘re’. It first replaces URLs by a tag ‘_url_’ and Twitter user tags by ‘_usernaem_’. It then replaces 11 types of emoticons with a tag, this is so that information from emoticons is not lost when the cleaning removes punctuation. Unnecessary punctuation is then removed and common English grammatical usages are corrected for to avoid the model to recount a word as unique simply because it has one of the grammatical tails attached to it.

***v.	pad_sentences***
This is a legacy function that has been superseded by pad_tweets. It performs the same function as pad_tweets, except that it would have to be called separately for both the annotated and the neutral data, while pad_tweets can perform padding for both annotated and neutral data together and return them separately.

***vi.	pad_tweets***
This function takes three arguments – the annotated data, the neutral data and the padding word (default = ‘<PAD/>’). It returns the padded data for both the annotated and the neutral data.
There is a method to define the ‘sequence_length’ as the maximum length for sequences in the data, however in practice this was found to be faulty as there would generally be at least one sequence which is extremely large reducing the significance of the rest of the data, hence after some study of the SAM tweet dataset it was decided that the ‘sequence_length’ would be specified as 40 for that dataset.
There is a function defined within this function called pad that takes in all the data and the required ‘sequence_length’, it then pads the data to the specified length. If the sequence is larger than the ‘sequence_length’ then it is truncated and returned.

***vii.	build_vocab***
This function takes as argument the sentences in the data, it then returns both a dictionary that maps from the words to an index and a list of words. It finds the unique words in the data in decreasing order of their frequency using a function ‘Counter’ from the python module ‘collections’.

***viii.	build_input_data***
This function takes three arguments – the sentences from the data, the labels and the dictionary that maps each word in the vocabulary to a unique integer. It returns two ‘numpy’ arrays – one with the data and the other with the labels. The model trains on ‘numpy’ arrays that have integer values for each word and this function helps build that input. It creates a ‘numpy’ array for the data by mapping each word to its corresponding integer value. It transforms the labels to a ‘numpy’ array as well.

***ix.	build_neutral_data***
This creates a numpy array the same as in the build_input_data function for the neutral data. It takes as input the neutral data and the dictionary that maps each word in the vocabulary to a unique integer. It returns the ‘numpy’ array for the neutral data.

***x.	batch_iter***
This is a legacy  function that has not been used at all. It might have been a more efficient way to load the data but it uses packages we are not used to, so we have just let it be and not spend much time on it.

### 4.	w2v
This module has one function namely train_word2vec. This takes as input the sentences as the ‘numpy’ array, the dictionary that maps the index to the words in the vocabulary, the number of features for the word embeddings (default = 300), the minimum words in set (default = 1) and the context window for the word embeddings (default = 10). It returns the embedding weights for the embedding layer.
The function first assigns a name to the word embedding model based on the values for the parameters and if that file exists it will load that model instead of creating a new one else it will create a new model. To do this it has the following parameters – number of workers = 4, down sampling setting for frequent words = 1e-3 and training algorithm set to skip-gram. It then saves the word2vec model in the ‘models’ folder.

## Auxiliary modules
There are 3 auxiliary modules

### 1.	classify_sentiment
This module calculates the sentiment in the form the probability (between 0 and 1) for a set of input sentences. If labels are available, it also calculates the accuracy on the whole set. To be able to do prediction for new sentences, each sentence first has to go through the same pre-processing steps as for the training set. The prediction is done in the function ‘label_tweets’. There is another function called ‘language_detection’ which is used to separate Dutch and English tweets. It takes as input a trained keras model <model>.h5, a vocabulary dictionary and a file with tweets to test and outputs probability and accuracy.

### 2.	extract_raw_data
This module is meant to extract the data from an annotated dataset named <dataset> and to split it into two files - <dataset>.pos (for the positively annotated set) and <dataset>.neg (for the negatively annotated set). 
It has one function named parseTweets that accepts as arguments the name of the dataset, the file path to the data and the fraction of tweets to be loaded. It has a parameter ‘numberOfMilestones’ that specifies how often to display information as the data is being loaded. The function then opens the required new files with the ‘open’ method, while it opens the data as a ‘pandas’ dataframe. It will go through the data and will write the data to the appropriate file based on the label column.

### 3.	init_model
This module is simply meant to load the ‘Keras’ model from a separate JSON file (for model architecture) and a h5 file (for model weights). It combines the information from both files into one keras model and returns the model.
The module has one function init that takes as input the file paths for the JSON file and for the h5 file. It returns the compiled model.
