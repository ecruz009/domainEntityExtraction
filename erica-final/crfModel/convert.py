"""
Code to convert *__output.txt to training and test files
No need to run unless training and test files are changed.
Returns tokenized terms with BIO notation to identify
keyphrases and offset boundaries.
Python version tested: 3.7.5
Prerequiresite:
* text and annotation files under ./medicalData/formatBIO/testBIO
* text and annotation files under ./medicalData/formatBIO/trainBIO
Output:
* files for CRF training under ./medicalData/convertedBIO/test
* files for CRF training under ./medicalData/convertedBIO/train
Usage:
$ python3 convert.py
Authorship:
Sagnik Choudhury 2017: initial version for JCDL 2017 paper
Agnese Chiatte 2017: initial version for JCDL 2017 paper
Gunnar Reiske 2019: for JCDL 2020 paper
Jian Wu 2020: for JCDL 2020 paper, add find_to_tag(), fixed bugs when parsing tokens containing punctuation. Add input arguments.
"""
import codecs
import os
import sys

#%%
#Made to accept formatBIO type files
# training corpus = original or larger
trainingCorpus = 'larger'
# trainingCorpus = 'original'


inputFileDir_test   = 'medicalData/formatBIO/' + trainingCorpus + '/testBIO/'
outputFileDir_test  = 'medicalData/convertedBIO/' + trainingCorpus + '/test/'
testListpath        = 'medicalData/' + trainingCorpus + '/testData/testList.txt'

inputFileDir_train  = 'medicalData/formatBIO/' + trainingCorpus + '/trainBIO/'
outputFileDir_train = 'medicalData/convertedBIO/' + trainingCorpus + '/train/'
trainListpath       = 'medicalData/' + trainingCorpus + '/trainData/trainList.txt'

#%%
print("working on test files: %s" % testListpath)
# tfs = codecs.open(testListpath,'r','utf-8-sig').read().split("\n")[:-1]
tfs = []

with codecs.open(testListpath,'r','utf-8-sig') as f:
    for line in f:
        tfs.append(line.strip('\r\n'))  # EC change: tfs.append(line.strip('\n'))  --> tfs.append(line.strip('\r\n')) 

for tf in tfs:
    print("processing: %s" % tf)
    # con = [x.split("\t") for x in codecs.open(inputFileDir_test+tf+"__output.txt",'r','utf-8-sig').read().split("\n")[:-1]]
    con = [x.split("\t") for x in codecs.open(inputFileDir_test+tf+"__output.txt",'r','utf-8-sig').read().split("\n")[:-1]]
    with codecs.open(outputFileDir_test+tf.split("_")[0]+".txt","w",'utf-8-sig') as f:
        for c in con:
             if len(c)==1:
                 f.write("\n")
             else:
                 token = c[0]
                 tag,left,right = c[1].strip().split()
                 f.write(token+'\t'+tag+'\t'+left+','+right+'\n')
print("%d test output to: %s" % (len(tfs), outputFileDir_test))


print("working on train files: %s" % trainListpath)
# EC change: tfs = codecs.open(trainListpath,'r','utf-8-sig').read().split("\n")[:-1]
tfs = codecs.open(trainListpath,'r','utf-8-sig').read().split("\r\n")[:-1]
for tf in tfs:
    con = [x.split("\t") for x in codecs.open(inputFileDir_train+tf+"__output.txt",'r','utf-8-sig').read().split("\n")[:-1]]
    with codecs.open(outputFileDir_train+tf.split("_")[0]+".txt","w",'utf-8-sig') as f:
        for c in con:
             if len(c)==1:
                 f.write("\n")
             else:
                 token = c[0]
                 tag,left,right = c[1].strip().split()
                 f.write(token+'\t'+tag+'\t'+left+','+right+'\n')
print("%d train output to: %s" % (len(tfs), outputFileDir_train))