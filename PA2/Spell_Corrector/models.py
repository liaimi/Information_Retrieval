
import sys
import os.path
import random
import itertools
import marshal
import time
#import gzip
from glob import iglob

unigram_count = {}
bygram_word_count = {}
trigram_word_count = {}
bygram_char_list = {}
sub_list= {}
del_list = {}
ins_list = {}
trans_list = {}
unigram_count_edit = {}
bygram_count_edit = {}
prob_sub_edit = {}
prob_del_edit = {}
prob_ins_edit = {}
prob_trans_edit = {}

alphabet = "abcdefghijklmnopqrstuvwxyz0123546789&$+_' "

# Fulfill the bygram indexing dictionary
def generate_bygrame(word):
    if (len(word) < 2): return
    global bygram_char_list
    for tup in itertools.izip(word[:-1], word[1:]):
        bygram = tup[0] + tup[1]
        if bygram not in bygram_char_list:
            newlist = [];
            newlist.append(word)
            bygram_char_list[bygram] = newlist
        else:
            if word not in bygram_char_list[bygram]:
                bygram_char_list[bygram].append(word)

# Process a line in the file, fulfill the ungram_word and bigram_word dict
def process_line(line):
    global unigram_count
    global bygram_word_count
    global bygram_char_list
    line = line.split(' ')
    if (len(line) is 0): return
    if (len(line) > 1):
        for tup in itertools.izip(line[:-1], line[1:]):
            if tup not in bygram_word_count:
                bygram_word_count[tup] = 1
            else:
                bygram_word_count[tup] += 1
            
            if tup[1] not in unigram_count:
                unigram_count[tup[1]] = 1
                generate_bygrame(tup[1])
            else:
                unigram_count[tup[1]] += 1
            
    if line[0] not in unigram_count:
        unigram_count[line[0]] = 1
        generate_bygrame(line[0])
    else:
        unigram_count[line[0]] += 1

#The funtion used to generate trigram_dict which would be used in stupid backoff method
#def process_line(line):
#    global unigram_count
#    global bygram_word_count
#    global bygram_char_list
#    line = line.split(' ')
#    if (len(line) is 0): return
#    if (len(line) > 2):
#        for tup in itertools.izip(line[:-2], line[1:-1], line[2:]):
#            if tup not in trigram_word_count:
#                trigram_word_count[tup] = 1
#            else:
#                trigram_word_count[tup] += 1
#            
#            if (tup[1],tup[2]) not in bygram_word_count:
#                bygram_word_count[(tup[1],tup[2])] = 1
#            else:
#                bygram_word_count[(tup[1],tup[2])] += 1
#
#            if tup[2] not in unigram_count:
#                unigram_count[tup[2]] = 1
#                generate_bygrame(tup[2])
#            else:
#                unigram_count[tup[2]] += 1
#                    
#    if(len(line) > 1):
#        if (line[0],line[1]) not in bygram_word_count:
#            bygram_word_count[(line[0],line[1])] = 1
#        else:
#            bygram_word_count[(line[0],line[1])] += 1
#        if line[1] not in unigram_count:
#            unigram_count[line[1]] = 1
#            generate_bygrame(line[1])
#        else:
#            unigram_count[line[1]] += 1
#
#    if line[0] not in unigram_count:
#        unigram_count[line[0]] = 1
#        generate_bygrame(line[0])
#    else:
#        unigram_count[line[0]] += 1

# Calculate the edi distance between two string, and detect what kinds of edit has been made
def edit_distance(s1, s2):
    global sub_list
    global del_list
    global ins_list
    global trans_list
    result = [];
    m = [([0] * (len(s2) + 1)) for i in range(len(s1) + 1)]
    for i in range(1, len(s1) + 1):
        m[i][0] = i
    for j in range(1, len(s2) + 1):
        m[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            k = 0
            if s1[i - 1] == s2[j - 1]: k = 0
            else: k = 1
            m[i][j] = min(m[i - 1][j - 1] + k, m[i - 1][j] + 1, m[i][j-1] + 1)
            if (i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]):
                m[i][j] = min(m[i][j], m[i-2][j-2] + k)
    index1 = len(s1)
    index2 = len(s2)
    while True:
        if index1 == 0 or index2 == 0:
            break
        if s1[index1 - 1] == s2[index2 - 1]: k = 0
        else: k = 1
        if (m[index1][index2] == m[index1-1][index2-1] and s1[index1-1] == s2[index2-1]):
            index1 = index1 - 1
            index2 = index2 - 1
        elif (m[index1][index2] == m[index1-1][index2-1] + 1 and s1[index1-1] != s2[index2-1]):
            if (s1[index1-1],s2[index2-1]) not in sub_list:
                sub_list[(s1[index1-1],s2[index2-1])] = 1
            else :
                sub_list[(s1[index1-1],s2[index2-1])] += 1
            index1 = index1 - 1
            index2 = index2 - 1
        elif m[index1][index2] == m[index1][index2-1] + 1:
            if (s1[index1-1],s2[index2-1]) not in ins_list:
                ins_list[(s1[index1-1],s2[index2-1])] = 1
            else :
                ins_list[(s1[index1-1],s2[index2-1])] += 1
            index2 = index2 - 1
        elif index1 > 1 and m[index1][index2] == m[index1-1][index2] + 1:
            if (s1[index1-2],s1[index1-1]) not in del_list:
                del_list[(s1[index1-2],s1[index1-1])] = 1
            else :
                del_list[(s1[index1-2],s1[index1-1])] += 1
            index1 = index1 - 1
        elif index1 > 1 and index2 > 1 and m[index1][index2] == m[index1-2][index2-2] + k:
            if (s1[index1-2],s1[index1-1]) not in trans_list:
                trans_list[(s1[index1-2],s1[index1-1])] = 1
            else :
                trans_list[(s1[index1-2],s1[index1-1])] += 1
            index1 = index1 - 2
            index2 = index2 - 2

# Scan the corpus and process each line
def scan_corpus(training_corpus_loc):
  for block_fname in iglob( os.path.join( training_corpus_loc, '*.txt' ) ):
    print >> sys.stderr, 'processing dir: ' + block_fname
    with open( block_fname ) as f:
      for line in f:
        line = line.rstrip('\n')
        process_line(line)

def read_edit1s(edit1s_loc):
  edit1s = []
  with open(edit1s_loc) as f:
    edit1s = [ line.rstrip().split('\t') for line in f if line.rstrip() ]
  return edit1s

def serialize_data(data, fname):
    with open(fname, 'wb') as f:
        marshal.dump(data, f)

# Process the edit1s.txt, fulfill the unigram_count_edit and bigram_count_edi
def process_edit1s(edit1s_loc):
    global unigram_count_edit
    global bygram_count_edit
    edit1s = read_edit1s(edit1s_loc)
    query_num = 1
    for (query,corrected_query) in edit1s:
        (query, corrected_query) = ('~' + query, '~' + corrected_query)
        for i in range(len(corrected_query)):
            if corrected_query[i] not in unigram_count_edit:
                unigram_count_edit[corrected_query[i]] = 1
            else:
                unigram_count_edit[corrected_query[i]] += 1
            if i < len(corrected_query)-1:
                if corrected_query[i:i+2] not in bygram_count_edit:
                    bygram_count_edit[corrected_query[i:i+2]] = 1
                else:
                    bygram_count_edit[corrected_query[i:i+2]] += 1
        query_num = query_num + 1
        edit_distance(corrected_query, query)

# Fulfill four channel model dictionary and calculate the probability
def build_edit():
    global unigram_count_edit
    global bygram_count_edit
    global sub_list
    global del_list
    global ins_list
    global trans_list
    global prob_sub_edit
    global prob_del_edit
    global prob_ins_edit
    global prob_trans_edit
    num_unigram = len(unigram_count_edit)
    num_bygram = len(bygram_count_edit)
    for key in sub_list:
        prob_sub_edit[key] = float(sub_list[key]+1)/(unigram_count_edit[key[1]]+len(alphabet))
    for key in ins_list:
        prob_ins_edit[key] = float(ins_list[key]+1)/(unigram_count_edit[key[0]]+len(alphabet))
    for key in del_list:
        prob_del_edit[key] = float(del_list[key]+1)/(bygram_count_edit[key[0]+key[1]]+len(alphabet)*len(alphabet))
    for key in trans_list:
        prob_trans_edit[key] = float(trans_list[key]+1)/(bygram_count_edit[key[0]+key[1]]+len(alphabet)*len(alphabet))


                           
if __name__ == '__main__':
    scan_corpus(sys.argv[1]);
    for k in bygram_char_list.keys():
        bygram_char_list[k].sort();
    serialize_data(unigram_count,'unigram')
    serialize_data(bygram_word_count,'bygram_word')
#    serialize_data(trigram_word_count,'trigram_word')
    serialize_data(bygram_char_list,'bygram_character')
    process_edit1s(sys.argv[2])
    serialize_data(unigram_count_edit,'uni_edit_dict')
    serialize_data(bygram_count_edit,'by_edit_dict')
    build_edit()
    serialize_data(prob_sub_edit,'sub.dict')
    serialize_data(prob_ins_edit,'ins.dict')
    serialize_data(prob_del_edit,'del.dict')
    serialize_data(prob_trans_edit,'trans.dict')

