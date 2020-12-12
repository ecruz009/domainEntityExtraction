import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle

from pprint import pprint

from sklearn_crfsuite import metrics

from DataExtraction import convertCONLLFormJustExtractionSemEvalPerfile
from FeatureExtraction import sent2labels,sent2features
from PhraseEval import phrasesFromTestSenJustExtractionWithIndex
import os
import glob
    
#%%  

'''Select dataset '''
trainingCorpus = 'original'
# trainingCorpus = 'larger'

'''Select model (optimized or unoptimized)'''
# optimized = 'unoptimized'
optimized = 'optimized'


if trainingCorpus == 'larger':
    os.chdir('medicalData/convertedBIO/larger/test/')
    testFiles = glob.glob('*.txt')
elif trainingCorpus == 'original':
    os.chdir('medicalData/convertedBIO/original/test/')
    testFiles = glob.glob('*.txt')
    
#%% 
os.chdir('../../../../')
for fileinLoc in testFiles:
    # fileinLoc2 = 'medicalData/convertedBIO/larger/test/' + fileinLoc
    # fileoutLoc = fileinLoc.split(".")[0]+".ann"
    #%% 
    # Experiment 1: Original Training Corpus + Original Model
    if trainingCorpus == 'original' and optimized == 'unoptimized':
        fileinLoc2 = 'medicalData/convertedBIO/original/test/' + fileinLoc
        fileoutLoc = fileinLoc.split(".")[0]+".ann"
        fileoutLoc = "medicalData/predictedANN/original/unoptimized/" + fileoutLoc
        crf = pickle.load(open("medicalData/original/unoptimized/linear-chain-crf.model.pickle", "rb"))
    #%% 
    # Experiment 2: Original Training Corpus + Optimized Model
    elif trainingCorpus == 'original' and optimized == 'optimized':
        fileinLoc2 = 'medicalData/convertedBIO/original/test/' + fileinLoc
        fileoutLoc = fileinLoc.split(".")[0]+".ann"        
        fileoutLoc = "medicalData/predictedANN/original/optimized/" + fileoutLoc
        crf = pickle.load(open("medicalData/original/optimized/linear-chain-crf.model.pickle", "rb"))        
    #%%     
    # Experiment 3: Larger Training Corpus + Optimized Model    
    elif trainingCorpus == 'larger' and optimized == 'unoptimized':
        fileinLoc2 = 'medicalData/convertedBIO/larger/test/' + fileinLoc
        fileoutLoc = fileinLoc.split(".")[0]+".ann"        
        fileoutLoc = "medicalData/predictedANN/larger/unoptimized/" + fileoutLoc
        crf = pickle.load(open("medicalData/larger/unoptimized/linear-chain-crf.model.pickle", "rb"))
    #%%     
    # Experiment 4: Larger Training Corpus + Original Model    
    elif trainingCorpus == 'larger' and optimized == 'optimized':
        fileinLoc2 = 'medicalData/convertedBIO/larger/test/' + fileinLoc
        fileoutLoc = fileinLoc.split(".")[0]+".ann"        
        fileoutLoc = "medicalData/predictedANN/larger/optimized/" + fileoutLoc
        crf = pickle.load(open("medicalData/larger/optimized/linear-chain-crf.model.pickle", "rb"))        
   
       
    #%%
    
    # crf = pickle.load(open("medicalData/linear-chain-crf.model.pickle", "rb"))
    # (test_sents,test_sents_indices) = convertCONLLFormJustExtractionSemEvalPerfile(fileinLoc)
    (test_sents,test_sents_indices) = convertCONLLFormJustExtractionSemEvalPerfile(fileinLoc2)
    
    # pprint(test_sents[0])
    
    X_test = [sent2features(s) for s in test_sents]
    
    # pprint(X_test)
    
    y_test = [sent2labels(s) for s in test_sents]
    # pprint(y_test)
    
    y_pred = crf.predict(X_test)
    # pprint(y_pred)
    
    if trainingCorpus == 'larger':
        labels = list(crf.classes_)
        labels.remove('O')
        labels.remove('1')
        labels.remove('3')
        labels.remove('8')
        labels.remove('9')
    
    elif trainingCorpus == 'original':
        labels = list(crf.classes_)
        labels.remove('O')
        
    # print(labels)
    sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))
    print((metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3)))
    
    test_sents_pls = []  #test sentences with predicted labels
    for index,testsent in enumerate(test_sents):
        sent=[]
        pls = y_pred[index]
        for (token,pl) in zip(testsent,pls):
            nt=(token[0],token[1],pl)
            sent.append(nt)
        test_sents_pls.append(sent)
    
    test_sents_pls_phrases=[phrasesFromTestSenJustExtractionWithIndex(x,y) for (x,y) in zip(test_sents_pls,test_sents_indices)]
    i=0
    with open(fileoutLoc,"w", encoding = 'utf8') as f:
        for sen in test_sents_pls_phrases:
            phrases=sen[-1]['phrases']
            for (p,pis,pie) in phrases:
                f.write("T{0}\tKEYPHRASE_NOTYPES {1} {2}\t{3}\n".format(str(i),pis,pie,p))
                i+=1
    print("classified file written at",fileoutLoc)
    print('')


