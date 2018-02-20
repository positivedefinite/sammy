import pandas as pd
import numpy as np

def parseTweets(dataset_name, fpath, percentage_of_tweets):
    '''
    Opens a tsv file containing the sentiment (first column) and tweet (second column)
    If you run out of memory, try editing this file and adding more milestones.
    '''
    numberOfMilestrones = 10
# how many times do you want to be informed of progress? 1=only final state reported, 2=halfway & end
    labelColumn = 1
    textColumn = 2
    
    file = pd.read_csv(fpath, sep='\t', encoding = 'utf-8',  header=None, error_bad_lines=False)
    file_pos = open('data/'+str(dataset_name)+'/'+str(dataset_name)+'.pos', "w", encoding = 'utf-8')
    file_neg = open('data/'+str(dataset_name)+'/'+str(dataset_name)+'.neg', "w", encoding = 'utf-8')
    #file_exp = open('data/'+str(dataset_name)+'/'+str(dataset_name)+'_exceptions.txt', "a")
    length = int(len(file)*percentage_of_tweets)
    milestone = int(length/numberOfMilestrones)
    exceptions = 0
    
    for i in range(0,length):
        if file[labelColumn][i]==1:
            file_pos.write(file[textColumn][i]+"\n")
        elif file[labelColumn][i]==0:
            file_neg.write(file[textColumn][i]+"\n")
        else:
            exceptions+=1#file_exp.write(file[textColumn][i]+"\n")
        if i%milestone==0:
            #open & reopen to save at every milestone
            file_pos.close()
            file_neg.close()
            #file_exp.close()
            file_pos = open('data/'+str(dataset_name)+'/'+str(dataset_name)+'.pos', "a", encoding = 'utf-8')
            file_neg = open('data/'+str(dataset_name)+'/'+str(dataset_name)+'.neg', "a", encoding = 'utf-8')
            #file_exp = open('data/'+str(dataset_name)+'/'+str(dataset_name)+'_exceptions.txt', "a")
            print('Parsed '+str(i)+r'/'+str(length)+' tweets with '+str(exceptions)+' exceptions.')
    file_pos.close()
    file_neg.close()
    #file_exp.close()
    print('Job finished. Go home and drink a beer.')