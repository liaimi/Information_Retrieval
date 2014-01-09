#!/bin/env python
from collections import deque
import os, glob, os.path
import sys
import re
from time import time

if len(sys.argv) != 2:
    print >> sys.stderr, 'usage: python query.py index_dir'
    os._exit(-1)

def popLeftOrNone(p):
    if len(p) > 0:
        posting = p.popleft()
    else:
        posting = None
    return posting

# This function merge two postings to one
def merge_posting (postings1, postings2):
    new_posting = []
    i = 0
    j = 0
    while (i < len(postings1) and j < len(postings2)):
        if (postings1[i] == postings2[j]):
            new_posting.append(postings1[i])
            i = i + 1
            j = j + 1
        else:
            if postings1[i] < postings2[j]:
                i = i + 1
            else: j = j + 1
    return new_posting


# file locate of all the index related files
index_dir = sys.argv[1]
index_f = open(index_dir+'/corpus.index', 'r')
word_dict_f = open(index_dir+'/word.dict', 'r')
doc_dict_f = open(index_dir+'/doc.dict', 'r')
posting_dict_f = open(index_dir+'/posting.dict', 'r')

word_dict = {}
doc_id_dict = {}
file_pos_dict = {}
doc_freq_dict = {}

print >> sys.stderr, 'loading word dict'
for line in word_dict_f.readlines():
    parts = line.split('\t')
    word_dict[parts[0]] = int(parts[1])
print >> sys.stderr, 'loading doc dict'
for line in doc_dict_f.readlines():
    parts = line.split('\t')
    doc_id_dict[int(parts[1])] = parts[0]
print >> sys.stderr, 'loading index'
for line in posting_dict_f.readlines():
    parts = line.split('\t')
    term_id = int(parts[0])
    file_pos = int(parts[1])
    doc_freq = int(parts[2])
    file_pos_dict[term_id] = file_pos
    doc_freq_dict[term_id] = doc_freq

# This function returns the posting lists of a given term_id
def read_posting(term_id):
    global index_f
    index_f.seek(file_pos_dict[term_id])
    line = index_f.readline()
#    parts = line.split(' ')
#    posting_list = parts[2].split(',');
    posting_list = line.split(',')
    posting_list = map(int, posting_list)
    return posting_list

# This function returns the index of the term_id in the word_id_list that have lowest frequency
def min_frequent_word(word_id_list):
    minfrequent = doc_freq_dict[word_id_list[0]]
    min = 0
    for index in range(len(word_id_list)):
        if (doc_freq_dict[word_id_list[index]] < minfrequent):
            min = index
            minfrequent = doc_freq_dict[word_id_list[index]]
    return min


# read query from stdin
while True:
    input = sys.stdin.readline()
    input = input.strip()
    if len(input) == 0: # end of file reached
        break

    input_parts = input.split()
    # translate words into word_ids
    word_id_list = [];
    unseenWordExists = False
    Final_posting = []
   
    for word in input_parts:
        if word_dict.has_key(word):
            word_id_list.append(word_dict[word])
        else:
            unseenWordExists = True
            break

    if not unseenWordExists:
        if len(word_id_list) == 1:
            Final_posting = read_posting(word_id_list.pop())
        else:
            first = True
            while len(word_id_list) > 0:
                if first:
                    first = False
                    word1 = word_id_list.pop(min_frequent_word(word_id_list))
                    word2 = word_id_list.pop(min_frequent_word(word_id_list))
                    post1 = read_posting(word1)
                    post2 = read_posting(word2)
                    Final_posting = merge_posting(post1, post2)
                else:
                    word = word_id_list.pop(min_frequent_word(word_id_list))
                    post = read_posting(word)
                    Final_posting = merge_posting(Final_posting, post)

        if len(Final_posting) is not 0:
            result_docs = []
            for doc_id in Final_posting:
                result_docs.append(doc_id_dict[doc_id])

            result_docs.sort()

            for doc_name in result_docs:
                print >> sys.stdout, doc_name
        else:
            print >> sys.stdout, 'no results found'
                
    else:
        print >> sys.stdout, 'no results found'

#
## you need to translate words into word_ids
## don't forget to handle the case where query contains unseen words
## next retrieve the postings list of each query term, and merge the posting lists
## to produce the final result
#
## posting = read_posting(word_id)
#
## don't forget to convert doc_id back to doc_name, and sort in lexicographical order
## before printing out to stdout
#
