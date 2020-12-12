import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#from itertools import chain

#import nltk
#import sklearn
import seaborn as sns
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import scipy
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from pprint import pprint
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np

from FeatureExtraction import sent2labels,sent2features
# from PhraseEval import phrasesFromTestSenJustExtraction,phrase_extraction_report
from DataExtraction import convertCONLLFormJustExtractionSemEval

def main(trainingCorpus):
    # trainingCorpus = 'larger'
    # trainingCorpus = 'original'
    trainFile = "medicalData/convertedBIO/" + trainingCorpus + "/combinedTrain.txt"
    testFile  = "medicalData/convertedBIO/" + trainingCorpus + "/combinedTest.txt"
    
    train_sents = convertCONLLFormJustExtractionSemEval(trainFile)
    test_sents = convertCONLLFormJustExtractionSemEval(testFile)
        
    # pprint(train_sents[0])
    # print('\n')
    # pprint(test_sents[0])
    # print('\n')
        
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
#%%
    c1 = 0.1
    c2 = 0.1
    
    crf = sklearn_crfsuite.CRF(\
    algorithm='lbfgs',\
    c1=c1,\
    c2=c2,\
    max_iterations=100,\
    all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    
    if trainingCorpus == 'larger':
        labels = list(crf.classes_)
        labels.remove('O')
        labels.remove('1')
        labels.remove('3')
        labels.remove('8')
        labels.remove('9')
        pickle.dump(crf,open("medicalData/larger/unoptimized/linear-chain-crf.model.pickle","wb"), protocol = 0, fix_imports = True)
    elif trainingCorpus == 'original':   
        labels = list(crf.classes_)
        labels.remove('O')
        pickle.dump(crf,open("medicalData/original/unoptimized/linear-chain-crf.model.pickle","wb"), protocol = 0, fix_imports = True)
    # print(labels)
    print('\n')
    
    y_pred = crf.predict(X_test)
    
    sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))

    print('\nTest Results (Original Model, Training Corpus: ' + trainingCorpus + ')')
    print('c1 = %s, c2 = %s' %(c1, c2))
    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    
    # y_predT = crf.predict(X_train)
    # print('\nTrain Results (Original Model, Training Corpus: ' + trainingCorpus + ')')
    # print('c1 = %s, c2 = %s' %(c1, c2))
    # print(metrics.flat_classification_report(y_train, y_predT, labels=sorted_labels, digits=3))
    
 #%%    
    # '''
    # define fixed parameters and parameters to search

    def report(results, n_top=15):
        print("")
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            print(candidates)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})"
                      .format(results['mean_test_score'][candidate],
                              results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
            
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)
        
    # params_space = {
    #     'c1': scipy.stats.expon(scale=0.5),
    #     'c2': scipy.stats.expon(scale=0.5),
    # }
        
    # search
    # rs = RandomizedSearchCV(crf, params_space,
    #                         cv=3,
    #                         verbose=1,
    #                         n_jobs=-1,
    #                         n_iter=10,
    #                         return_train_score=True,
    #                         scoring=f1_scorer)
    
    param_grid = {
        'c1': [0.10, 0.20, 0.30, 0.40],
        'c2': [0.10, 0.20, 0.30, 0.40],
        }

    numSplits = 5
    rs = GridSearchCV(crf, param_grid,
                      cv = numSplits,
                      verbose = -1,
                      n_jobs=-1,
                      return_train_score = True,
                      scoring = f1_scorer)
    
    rs.fit(X_train, y_train)
    
    report(rs.cv_results_)
    
    # print(rs.cv_results_['params'])
    # print(rs.cv_results_['split0_test_score'])
    # print(rs.cv_results_['split1_test_score'])
    # print(rs.cv_results_['split2_test_score'])
    # print(rs.cv_results_['split3_test_score'])
    # print(rs.cv_results_['split4_test_score'])
    
    split1 = (np.array([rs.cv_results_['split0_test_score']])).T
    split2 = (np.array([rs.cv_results_['split0_test_score']])).T
    split3 = (np.array([rs.cv_results_['split0_test_score']])).T
    split4 = (np.array([rs.cv_results_['split0_test_score']])).T
    split5 = (np.array([rs.cv_results_['split0_test_score']])).T
    splitCompiled = np.hstack((split1, split2, split3, split4, split5))
        
    _x = [s['c1'] for s in rs.cv_results_['params']]
    _y = [s['c2'] for s in rs.cv_results_['params']]
    _c = [s for s in rs.cv_results_['mean_test_score']]
        
    fig = plt.figure()
    fig.set_size_inches(3,3)
    ax = plt.gca()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    # ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(min(_c), max(_c)))
    ax.set_title("Hyperparameter Gridsearch CV Results \n (min={:0.3}, max={:0.3})".format(min(_c), max(_c)))
    

    sc = ax.scatter(_x, _y, c=_c, s=60, alpha=0.9)
    
    # plt.text(0.1, 0.11,'Unoptimized Model', fontsize=10, horizontalalignment='left',verticalalignment='bottom')
    # if trainingCorpus == 'larger':
    #     plt.text(0.1, 0.39,'Optimized Model', fontsize=10, horizontalalignment='left',verticalalignment='top')
    # elif trainingCorpus == 'original': 
    #     plt.text(0.1, 0.2,'Optimized Model', fontsize=10, horizontalalignment='left',verticalalignment='top')

    plt.colorbar(sc)
    if trainingCorpus == 'larger':
        plt.savefig('medicalData/larger/optimized/OptimizedLarger.png')
    elif trainingCorpus == 'original': 
        plt.savefig('medicalData/original/optimized/OptimizedOriginal.png')

        
    # '''
    
    crf_best = rs.best_estimator_
    if trainingCorpus == 'larger':
        pickle.dump(crf_best,open("medicalData/larger/optimized/linear-chain-crf.model.pickle","wb"), protocol = 0, fix_imports = True)
    elif trainingCorpus == 'original':
        pickle.dump(crf_best,open("medicalData/original/optimized/linear-chain-crf.model.pickle","wb"), protocol = 0, fix_imports = True)
        
    # print('best params:', rs.best_params_)
    # print('best CV score:', rs.best_score_)
    # print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))   
    
    y_pred = crf_best.predict(X_test)
    print('\nTest Results (Optimized Model, Training Corpus: ' + trainingCorpus + ')')
    # print('c1 = %s, c2 = %s' %(c1, c2))
    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    
    # y_predT = crf_best.predict(X_train)
    # print('\nTrain Results (Optimized Model, Training Corpus: ' + trainingCorpus + ')')
    # # print('c1 = %s, c2 = %s' %(c1, c2))
    # print(metrics.flat_classification_report(y_train, y_predT, labels=sorted_labels, digits=3))
        
    #%%
    '''
    x = np.linspace(1,16,num=16)
    fig = plt.figure()
    fig.set_size_inches(10,6)
    ax = plt.gca()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel('Regularization Coefficients')
    ax.set_ylabel('Split Test Score')
    ax.set_title("Randomized Hyperparameter Search CV Results (min={:0.3}, max={:0.3})".format(min(_c), max(_c)))
    ax.set_title("Hyperparameter Gridsearch CV Results (min={:0.3}, max={:0.3})".format(min(_c), max(_c)))
    # plt.xticks([1, 2])
    ax.set_xticklabels(['{0.1, 0.1}','{0.1, 0.2}','{0.1, 0.3}','{0.1, 0.4}',
                        '{0.2, 0.1}','{0.2, 0.2}','{0.2, 0.3}','{0.2, 0.4}',
                        '{0.3, 0.1}','{0.3, 0.2}','{0.3, 0.3}','{0.3, 0.4}',
                        '{0.4, 0.1}','{0.4, 0.2}','{0.4, 0.3}','{0.4, 0.4}'],rotation = 45)
    
    # ax.scatter(_x, _y, c=_c, s=60, alpha=0.9, edgecolors=[0,0,0])
    ax.scatter(x, split1, c='midnightblue', s=60, alpha=0.9, marker = 'o', label = 'Split 1')
    ax.scatter(x, split2, c='blue',         s=60, alpha=0.9, marker = '*', label = 'Split 2')
    ax.scatter(x, split3, c='steelblue',    s=60, alpha=0.9, marker = 's', label = 'Split 3')
    ax.scatter(x, split4, c='skyblue',  s=60, alpha=0.9, marker = '+', label = 'Split 4')
    ax.scatter(x, split5, c='turquoise',    s=60, alpha=0.9, marker = 'D', label = 'Split 5')
    
    # ax.scatter(x, split1, c= split1, s=60, alpha=0.9, marker = 'o', label = 'Split 1')
    # ax.scatter(x, split2, c= split2, s=60, alpha=0.9, marker = '*', label = 'Split 2')
    # ax.scatter(x, split3, c= split3, s=60, alpha=0.9, marker = 's', label = 'Split 3')
    # ax.scatter(x, split4, c= split4, s=60, alpha=0.9, marker = '+', label = 'Split 4')
    # ax.scatter(x, split5, c= split5, s=60, alpha=0.9, marker = 'D', label = 'Split 5')
    ax.legend()
    ax.grid(True)
    # sc = plt.scatter(xy, xy, c=z, vmin=0, vmax=20, s=35, cmap=cm)
    # plt.colorbar(sc)
    '''
    return splitCompiled
    
#%%
    # sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))

    # print('\nTest Results (Training Corpus: ' + trainingCorpus + ')')
    # print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    
    # print('\nTrain Results (Training Corpus: ' + trainingCorpus + ')')
    # y_pred = crf.predict(X_train)
    # print(metrics.flat_classification_report(y_train, y_pred, labels=sorted_labels, digits=3))
    
if __name__ == "__main__":
    originalSplits = main('original')
    # largerSplits = main('larger')