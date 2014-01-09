#!/bin/env python
from collections import deque
from operator import itemgetter
import os, glob, os.path
import sys
import re

if len(sys.argv) != 3:
    print >> sys.stderr, 'usage: python index.py data_dir output_dir'
    os._exit(-1)

total_file_count = 0
root = sys.argv[1]
out_dir = sys.argv[2]
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# this is the actual posting lists dictionary
# word id -> {position_in_file, doc freq}
posting_dict = {}
# this is a dict holding document name -> doc_id
doc_id_dict = {}
# this is a dict holding word -> word_id
word_dict = {}
# this is a queue holding block names, later used for merging blocks
block_q = deque([])

# function to count number of files in collection
def count_file():
    global total_file_count
    total_file_count = total_file_count + 1

def print_list(list):
    result = ''
    for item in list:
        result = result + str(item) + ','
    return result[:-1]

# function for printing a line in a postings list to a given file
def print_posting(file, posting_line, term_count, term):
    file.write(str(term) + ' ')
    file.write(str(term_count) + ' ')
    file.write(print_list(posting_line))
    file.write('\n')

def popLeftOrNone(p):
    if len(p) > 0:
        posting = p.popleft()
    else:
        posting = None
    return posting

# function for merging two lines of postings list to create a new line of merged results
def merge_posting (line1, line2):
    answer = []
    psts1 = line1.split(' ')
    psts2 = line2.split(' ')
    list1 = psts1[2].split(',')
    list2 = psts2[2].split(',')
    posting_list_1 = deque([])
    posting_list_2 = deque([])
    for l in list1:
        posting_list_1.append(int(l.strip()))
    for l in list2:
        posting_list_2.append(int(l.strip()))
    pl1 = popLeftOrNone(posting_list_1)
    pl2 = popLeftOrNone(posting_list_2)
    while pl1 is not None and pl2 is not None:
        if pl1 == pl2:
            answer.append(pl1)
            pl1 = popLeftOrNone(posting_list_1)
            pl2 = popLeftOrNone(posting_list_2)
        elif pl1 < pl2:
            answer.append(pl1)
            pl1 = popLeftOrNone(posting_list_1)
        else:
            answer.append(pl2)
            pl2 = popLeftOrNone(posting_list_2)
    while pl1 is not None:
        answer.append(pl1)
        pl1 = popLeftOrNone(posting_list_1)
    while pl2 is not None:
        answer.append(pl2)
        pl2 = popLeftOrNone(posting_list_2)
    return psts1[0]+ ' '+str(int(psts1[1])+int(psts2[1]))+' '+print_list(answer)


doc_id = -1
word_id = 0

for dir in sorted(os.listdir(root)):
    print >> sys.stderr, 'processing dir: ' + dir
    dir_name = os.path.join(root, dir)
    block_pl_name = out_dir+'/'+dir
    # append block names to a queue, later used in merging
    block_q.append(dir)
    block_pl = open(block_pl_name, 'w')
    term_doc_list = []

    for f in sorted(os.listdir(dir_name)):
        count_file()
        file_id = os.path.join(dir, f)
        doc_id += 1
        doc_id_dict[file_id] = doc_id
        fullpath = os.path.join(dir_name, f)
        file = open(fullpath, 'r')
        for line in file.readlines():
            tokens = line.strip().split()
            for token in tokens:
                if token not in word_dict:
                    word_dict[token] = word_id
                    word_id += 1
                term_doc_list.append( (word_dict[token], doc_id) )
    print >> sys.stderr, 'sorting term doc list for dir:' + dir
    # sort the term_doc_list based on termID   
    term_doc_list.sort(key=itemgetter(0))
    prev = term_doc_list[0][0]
    posting_line = []
    term_count = 0
    for term_doc in term_doc_list:
        if term_doc[0] == prev:
            term_count = term_count +1
            if term_doc[1] not in posting_line:
                posting_line.append(term_doc[1])
        else:
            print_posting(block_pl, posting_line, term_count, prev)
            posting_line = []
            posting_line.append(term_doc[1])
            prev = term_doc[0]
            term_count = 1
    print_posting(block_pl, posting_line, term_count, prev)
    block_pl.close()
print >> sys.stderr, '######\nposting list construction finished!\n##########'

print >> sys.stderr, '\nMerging postings...'
isPrint = False;
while True:
    if len(block_q) <= 1:
        break
    if len(block_q) == 2:
        print >> sys.stderr, '\nfinal two block merge'
        isPrint = True;
    b1 = block_q.popleft()
    b2 = block_q.popleft()
    print >> sys.stderr, 'merging %s and %s' % (b1, b2)
    b1_f = open(out_dir+'/'+b1, 'r')
    b2_f = open(out_dir+'/'+b2, 'r')
    comb = b1+'+'+b2
    comb_f = open(out_dir + '/'+comb, 'w')
    # merging the two blocks of posting lists
    # write the new merged posting lists block to file 'comb_f'
    line1 = b1_f.readline()
    line2 = b2_f.readline()
    while len(line1) is not 0 and len(line2) is not 0:
        line1_sp = line1.split(' ')
        line2_sp = line2.split(' ')
        word1 = int(line1_sp[0].strip())
        word2 = int(line2_sp[0].strip())
        freq1 = int(line1_sp[1].strip())
        freq2 = int(line2_sp[1].strip())
        if word1 == word2:
            pos_merge = merge_posting(line1,line2)
            if isPrint:
                posting_dict[word1] = (comb_f.tell(), freq1+freq2)
                comb_f.write(pos_merge.split(' ')[2] + '\n')
            else:
                comb_f.write(pos_merge + '\n')
            line1 = b1_f.readline()
            line2 = b2_f.readline()
        elif word1 < word2:
            if isPrint:
                posting_dict[word1] = (comb_f.tell(), freq1)
                comb_f.write(line1_sp[2])
            else:
                comb_f.write(line1)
            line1 = b1_f.readline()
        else:
            if isPrint:
                posting_dict[word2] = (comb_f.tell(), freq2)
                comb_f.write(line2_sp[2])
            comb_f.write(line2)
            line2 = b2_f.readline()
    while len(line1) is not 0:
        line1_sp = line1.split(' ')
        word1 = int(line1_sp[0].strip())
        freq1 = int(line1_sp[1].strip())
        if isPrint:
            posting_dict[word1] = (comb_f.tell(), freq1)
            comb_f.write(line1_sp[2])
        else:
            comb_f.write(line1)
        line1 = b1_f.readline()
    while len(line2) is not 0:
        line2_sp = line2.split(' ')
        word2 = int(line2_sp[0].strip())
        freq2 = int(line2_sp[1].strip())
        if isPrint:
            posting_dict[word2] = (comb_f.tell(), freq2)
            comb_f.write(line2_sp[2])
        else:
            comb_f.write(line2)
        line2 = b2_f.readline()
    b1_f.close()
    b2_f.close()
    comb_f.close()
    os.remove(out_dir+'/'+b1)
    os.remove(out_dir+'/'+b2)
    block_q.append(comb)

print >> sys.stderr, '\nPosting Lists Merging DONE!'


# rename the final merged block to corpus.index
final_name = block_q.popleft()
os.rename(out_dir+'/'+final_name, out_dir+'/corpus.index')

# print all the dictionary files
doc_dict_f = open(out_dir + '/doc.dict', 'w')
word_dict_f = open(out_dir + '/word.dict', 'w')
posting_dict_f = open(out_dir + '/posting.dict', 'w')
print >> doc_dict_f, '\n'.join( ['%s\t%d' % (k,v) for (k,v) in sorted(doc_id_dict.iteritems(), key=lambda(k,v):v)])
print >> word_dict_f, '\n'.join( ['%s\t%d' % (k,v) for (k,v) in sorted(word_dict.iteritems(), key=lambda(k,v):v)])
print >> posting_dict_f, '\n'.join(['%s\t%s' % (k,'\t'.join([str(elm) for elm in v])) for (k,v) in sorted(posting_dict.iteritems(), key=lambda(k,v):v)])
doc_dict_f.close()
word_dict_f.close()
posting_dict_f.close()

print total_file_count


    