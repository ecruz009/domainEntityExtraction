"""
Conversion script expecting data in Semeval2017 format
for keyphrase extraction (grouped as txts and ann files)
Returns tokenized terms with BIO notation to identify
keyphrases and offset boundaries.
Python version tested: 3.7.5
Prerequiresite: 
* text and annotation files under ./medicalData/testData
* text and annotation files under ./medicalData/trainData
Output: 
* intermediate files for CRF training under ./medicalData/formatBIO/testBIO
* intermediate files for CRF training under ./medicalData/formatBIO/trainBIO
Usage: 
$ python3 semeval_to_BIO.py 
examplle: 
$ python3 semeval_to_BIO.py -i medicalData/trainData -o medicalData/formatBIO/trainBIO
Authorship:
Sagnik Choudhury 2017: initial version for JCDL 2017 paper
Agnese Chiatte 2017: initial version for JCDL 2017 paper
Gunnar Reiske 2019: for JCDL 2020 paper
Jian Wu 2020: for JCDL 2020 paper, add find_to_tag(), fixed bugs when parsing tokens containing punctuation. Add input arguments.
"""

import os
import string
import re
import codecs
import argparse
import glob, os, os.path


#%%
# find the bio tag of the current token, the span of the token is provided
def find_bio_tag(current_off,curr_end,kpw_starts,kpw_offs):
    # determine the correct tag and write the token
    if current_off in kpw_starts:
         bio_tag = 'B-KP'
    else:
         inner = False
         for start, end in kpw_offs.keys():
             if (current_off > start) & (curr_end <= end):
                 inner = True
                 break
         if inner:
              bio_tag = 'I-KP'
         else:
              bio_tag = 'O'
    return bio_tag

#%%
# judge if a token is the last word in a sentence
# cases(things like "B. burgdorferi", "USA).", "Borrelia spp. [followed by a starting sentence]" will not end a sentence)
def is_sent_end(ti,token,tokens):
    try: # check if the next word starts with a capital letter
        if re.match(r"\w+\.$",token) and re.match(r"[^A-Z]",tokens[ti+1][0]):
            return False
    except IndexError: # the end of paragraph, so must be the end of sentence
        return True

    # single letter like "I. Borrelia"
    if re.match(r".*[A-Z]\.$",token):
        return False
    elif token[-1] == '.':
        return True
#%%
parser = argparse.ArgumentParser()

# training corpus = original or larger
# trainingCorpus = 'larger'
trainingCorpus = 'original'

# dataset = test or train
# dataset = 'test'
dataset = 'train'

defaultInputString     = 'medicalData/' + trainingCorpus + '/' + dataset + 'Data'
defaultOutputputString = 'medicalData/formatBio/' + trainingCorpus + '/' + dataset + 'BIO'

parser.add_argument("-i","--input_root",metavar="INPUT_ROOT",type=str,default=defaultInputString,help="Relative path of input directory, containing anns/ and txts/. Default is medicalData/trainData.")
parser.add_argument("-o","--output_root",metavar="OUTPUT_ROOT",type=str,default=defaultOutputputString,help="Relative path of output directory. Default is medicalData/formatBIO/trainBIO.")
      
args = parser.parse_args()

#BIO converion for all testData
input_root = args.input_root
anns_folder = 'anns'
txt_folder = 'txts'

#Lines will be added in append mode, so if output_fold is non-empty,
# make sure to remove everything in it first

output_fold = args.output_root
# filelist = glob.glob(os.path.join(mydir, "*.bak"))
# for f in filelist:
#     os.remove(f)

txts = sorted( [os.path.join(input_root, txt_folder, txt) for txt in os.listdir(os.path.join(input_root, txt_folder))])
anns = sorted([os.path.join(input_root, anns_folder, ann) for ann in os.listdir(os.path.join(input_root, anns_folder))])
#Kept same naming convention as original experiments
outs = sorted( [os.path.join(output_fold, txt[:-4]+'__output.txt') for txt in os.listdir(os.path.join(input_root, txt_folder))])
#Removes the hidden file .DS_Store from list
try:
    txts.remove(os.path.join(input_root, txt_folder, ".DS_Store"))
    anns.remove(os.path.join(input_root, anns_folder, ".DS_Store"))
    outs.remove(os.path.join(output_fold, ".DS_Store"[:-4]+'__output.txt'))
except:
    pass


for txt, ann, out in zip(txts, anns, outs):

    current_off = 0
    kpw_starts={}
    kpw_offs={}

    print("processing %s " % txt)
    with codecs.open(txt,'r','utf-8-sig') as txtf, codecs.open(ann,'r','utf-8-sig') as annf, codecs.open(out, 'w','utf-8-sig') as outf:

        tokens = txtf.read().split()
        # Takes the actual keyphrases from the ann file and offsets
        ann_lines =annf.readlines()

        for line in ann_lines:

            #Skip semantic relations, if any
            if line[0]=='R' or line[0]=="*":
                continue

            offs = line.split('\t')[1].split()
            start = offs[1]

            if len(offs)== 3:
                #Most of cases
                end = line.split('\t')[1].split()[2]

            # Some of semeval annotations have intermediate/nested offsets, but here we only
            # consider start and outermost end
            elif len(offs)==4:

                end = line.split('\t')[1].split()[3]

            kp = line.split('\t')[2][:-1]


            kpw_starts[int(start)] = dict(end=end, kp=kp)
            kpw_offs[(int(start), int(end))] = kp


        """
        OutFormat:
        one token per line
        one empty newline in between sentences
        token <tab> (BIO)-KP <space> start_offset <space> end_offset <\n>
        
        """

        for ti,token in enumerate(tokens):

            curr_end = current_off + len(token)
            # if there is a single punctuation mark, such as "-", treat it specially
            if re.match(r"[-<>&=]",token):
                m = None
            else:
                # separate punctuation marks with alphenumeric characters
                m = re.match(r"^(\W*)([\w-]*)(\W*$)",token)

            if m: 
                punct_pre,token_word,punct_suf = tokens_punct_match = m.groups()
                # the prefix puncutation, if exists
                for punc in punct_pre:
                    outf.write(punc+'\t'+' O '+str(current_off)+' '+str(current_off+1)+'\n')
                    current_off = current_off + 1

                # determine the correct tag and write the token 
                bio_tag = find_bio_tag(current_off,current_off+len(token_word),kpw_starts,kpw_offs)
                outf.write(token_word+'\t'+bio_tag+' '+str(current_off)+' '+str(current_off+len(token_word))+'\n')
                current_off = current_off+len(token_word)
            
                # the last subtoken, if the token matches something like "B. burgdorferi", "." will be marked as "KP-I"
                for punc in punct_suf:
                    bio_tag = find_bio_tag(current_off,current_off+1,kpw_starts,kpw_offs)
                    outf.write(punc+'\t'+' '+bio_tag+' '+str(current_off)+' '+str(current_off+1)+'\n')
                    current_off = current_off + 1
            else: 
                print("warning: cannot find a match: %s " % token)
                # if the subtoken looks like "[12,13].", we should segment it 
                m = re.match(r"(.*\d+\])(\.)$",token)
                if m: 
                    sub_tokens = m.groups()
                else:
                    # delimit the string by '/', e.g., 'and/or', if it exists
                    sub_tokens = re.findall(r'[^\\/]+|[\\/]', token)
                for i,st in enumerate(sub_tokens):
                    bio_tag = find_bio_tag(current_off,current_off+len(st),kpw_starts,kpw_offs)
                    outf.write(st+'\t'+bio_tag+' '+str(current_off)+' '+str(current_off+len(st))+'\n')
                    current_off = current_off + len(st)
                    """
                    if i < len(sub_tokens)-1:
                        outf.write('/'+'\t'+' O '+str(current_off)+' '+str(current_off+1)+'\n')
                        current_off = current_off + 1
                    """

            # add a new line after a sentence 
            if is_sent_end(ti,token,tokens):
                outf.write('\n')
            current_off = curr_end +1 # move by a space 

            """
            try: # check if the next word starts with a capital letter
                if re.match(r"\w+\.$",token) and re.match(r"^[A-Z]",tokens[ti+1][0]):
                    pass
            except IndexError:
                pass
            if re.match(r".*[A-Z]\.$",token): 
                pass
            elif token[-1] == '.':
                outf.write("\n")
            """
            

print("\nprocessed %d files under: %s" % (len(txts),input_root))
print("output files to: %s" % output_fold)

