import glob
import os
import os.path

def ListFiles(dir,name):
    os.chdir(dir)
    myFiles = glob.glob('*.txt')
    lst = []
    for x in myFiles:
        lst.append(os.path.splitext(x)[0])
    
    os.chdir('../')
    with open(name + 'List.txt', 'w') as f:
        for item in lst:
            f.write("%s\n" % item)
    
    numFiles = len(lst)
    print('\n%s files combined into %sList in %sData' % (numFiles,name,name))            
            
    os.chdir('../../../')
    return lst

# training corpus = original or larger
trainingCorpus = 'larger'
# trainingCorpus = 'original'

   
test_dir  = 'medicalData/' + trainingCorpus + '/testData/txts/'
train_dir = 'medicalData/' + trainingCorpus + '/trainData/txts/'

test  = ListFiles(test_dir, 'test')
train = ListFiles(train_dir, 'train')
      
