import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import nltk
# nltk.download()
from nltk import pos_tag

def convertCONLLFormJustExtractionSemEval(loc):
    dT=open(loc, encoding = 'utf8').read().split("\n")[:-2]
    sI = [-1] + [i for i, x in enumerate(dT) if not x.strip()] + [len(dT)]
    sT1s = [dT[sI[i]+1:sI[i+1]] for i in range(len(sI)-1)]
    sTs = []
    for s in sT1s:
        ts= [(x.split("\t")[0],x.split("\t")[1]) for x in s]
        tokens = [(x[0],y[1],x[1]) for (x,y) in zip(ts,pos_tag([x[0] for x in ts])) ]
        tokens = [(x,y,z[0]) for (x,y,z) in tokens]
        sTs.append(tokens)
    return sTs

def convertCONLLFormJustExtractionSemEvalPerfile(loc):
    dT=open(loc, encoding = 'utf8').read().split("\n")[:-2]
    sI = [-1] + [i for i, x in enumerate(dT) if not x.strip()] + [len(dT)]
    sT1s = [dT[sI[i]+1:sI[i+1]] for i in range(len(sI)-1)]
    sTs = []
    sTIs = []
    for s in sT1s:
        ts= [(x.split("\t")[0],x.split("\t")[1],x.split("\t")[2]) for x in s]
        tss = [(x[0],y[1],x[1],x[2]) for (x,y) in zip(ts,pos_tag([x[0] for x in ts])) ]
        tokens = [(x,y,z[0]) for (x,y,z,w) in tss]
        tokenindices = [w for (x,y,z,w) in tss]
        sTs.append(tokens)
        sTIs.append(tokenindices)
    return (sTs,sTIs)