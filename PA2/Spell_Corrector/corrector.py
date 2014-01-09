#!/bin/env python
from collections import deque
from operator import itemgetter
import os, glob, os.path
import sys
import re
import operator
import itertools
import marshal
from math import log
from math import fabs
import time

queries_loc = 'data/queries.txt'
gold_loc = 'data/gold.txt'
google_loc = 'data/google.txt'

alphabet = "abcdefghijklmnopqrstuvwxyz0123546789&$+_' "

kgram_dict = {}
bygram_dict = {}
unigram_dict = {}
trigram_dict = {}
total_term_num = 0
lam = 0.2
prob_equal = 0.9
prob_edit = 0.01
jaccard_threshold_short = 0.5
jaccard_threshold_long = 0.5
valid_alphabet = ""
valid_two_char_word = [];
model_dict = [{},{},{},{}]
uni_edit_dict = {}
by_edit_dict = {}


def unserialize_data(fname):
    with open(fname, 'rb') as f:
        return marshal.load(f)

# Find all the one character words in dictionary
def find_valid_alphabet():
    global valid_alphabet
    global alphabet
    tmp = list(alphabet)
    for char in tmp:
        if char in unigram_dict:
            valid_alphabet += char

# Generate all the queries that is 1 edit distance away from word
def edits1(word):
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in splits for c in valid_alphabet if b]
    inserts    = [a + c + b     for a, b in splits for c in valid_alphabet]
    return list(deletes + transposes + replaces + inserts)

# Take in a string, check whether all the unigram and bygrams exist in dictionary
def allExists(query):
    words = query.split()
    if(words[0] not in unigram_dict): return False
    for tup in itertools.izip(words[:-1], words[1:]):
        if tup not in bygram_dict: return False
        if tup[1] not in unigram_dict: return False
    return True

# Take in a list of string ,check whether all the unigrams exists in dictionary
def allUniExists(words):
    for word in words:
        if word not in unigram_dict: return False
    return True

# Generate all the candidates that is 1 edit distance away from query, and all it's word exist in dictionary
def allOneEditValidCandidate(query):
    candidates = edits1(query)
    result = []
    for candidate in candidates:
        candidate = candidate.strip().split()
        if (allUniExists(candidate)): result.append(candidate)

    return result

# Count for the number of bygram that exist in dictionary of a query
def rough_score(query):
    u_score = 0;
    b_score = 0;
    query = query.split()
    length = len(query)

    if query[0] in unigram_dict: u_score += 1
    
    if length > 1:
        for tup in itertools.izip(query[:-1], query[1:]):
            if tup in bygram_dict: b_score += 1
            if tup[1] in unigram_dict: u_score += 1
        return float(b_score)/(length - 1)

# Generate all the possible split and combine
def split_candidates(query):
    candidates = [query]
    words = query.split()
    for i in range(len(words)):
        if words[i]  not in unigram_dict:
            word = words[i]
            if len(word) > 6:
                splits = [(word[:k], word[k:]) for k in range (1,len(word))]
                for tup in splits:
                    if tup in bygram_dict:
                        candidate = words[:i] + list(tup) + words[i+1:]
                        tmp = ' '.join(candidate)
                        candidates.append(tmp)

    for i in range(len(query)):
        if query[i] is ' ':
            newQ = query[:i] + query[i+1:]
            if allExists(newQ): candidates.append(newQ)
    return candidates

# Find the split/combine that has highest rough score
def joint_split_candidates(query):
    max_candidate = '';
    max_score = float("-inf")
    for newQ in split_candidates(query):
        score = rough_score(newQ)
        if (score > max_score):
            max_score = score
            max_candidate = newQ
    return max_candidate

def getValues(word_list, index):
    result = []
    for i in range(0,len(index)):
        if index[i]!= -1:
            result.append(word_list[i][index[i]])
        else:
            result.append('~')
    return result;

def jaccard(word,compare,num):
    return float(num)/(len(word)+len(compare)-2-num)

# Generate all candidates for a certain word using bigram indexing
def generateCandidateWord(word, kgram_dict):
    result = []
    ed_list = []
    index = [0] * (len(word) - 1)
    word_list = []
    if (len(word) == 1): return [i for i in list(valid_alphabet)]
    for i in range(0, len(word) - 1):
        word_list.append(kgram_dict[word[i:i+2]])
    while True:
        if len([item for item in range(len(index)) if index[item] == -1]) == (len(word) - 1):
            break;
        cur = getValues(word_list,index)
        min_index, min_value = min(enumerate(cur), key = operator.itemgetter(1))
        equalList = [item for item in range(len(cur)) if cur[item] == min_value]

        threshold = 0
        if len(word) > 5: threshold = jaccard_threshold_long
        else: threshold = jaccard_threshold_short
        if jaccard(word,min_value,len(equalList)) > threshold:
            result.append(min_value)

        for i in equalList:
            if index[i] == len(word_list[i])-1:
                index[i] = -1
            elif index[i] != -1:
                index[i] += 1
    return result;

# Generate all candidates for a query
def generateCandidate(query):
    start = time.time()
    words = query.split()
    query_list = []
    candidate_query = []
    index1 = [0] * (len(words))
    index2 = [0] * (len(words))
    index3 = [1] * (len(words))
    for i in range(len(words)):
        if words[i] not in unigram_dict:
            index1[i] = 1
    for i in range(len(words)-2):
        if ((words[i],words[i+1]) not in bygram_dict):
            index2[i] = 1
    if ((words[len(words)-2],words[len(words)-1]) not in bygram_dict):
        index2[len(words)-2] = 1
        index2[len(words)-1] = 1
    for i in range(len(words)):
        if index1[i] == 1:
            if (i > 0): index3[i-1] = 0
            if (i < len(words)-1): index3[i+1] = 0
    index = [index1[i] or index2[i] and index3[i] for i in range(len(words))]

    have_list_index = [i for i in range(len(index)) if index[i] is 1]
    if len(have_list_index) > 2:
        combine_pair = list(itertools.combinations(have_list_index,2))
        for tup in combine_pair:
            q_list = []
            for i in range(len(words)):
                if (i not in tup): q_list.append([words[i]])
                else: q_list.append(generateCandidateWord(words[i],kgram_dict))
            candidate_query.extend(list(itertools.product(*q_list)))
    else:
        for i in range(len(words)):
            if (index[i] == 0):
                query_list.append([words[i]])
            else:
                query_list.append(generateCandidateWord(words[i],kgram_dict))
        candidate_query = list(itertools.product(*query_list))
    
    candidate_query.extend(allOneEditValidCandidate(query))
    return candidate_query

                              

def read_query_data():
  gold = []
  google = []
  with open(gold_loc) as f:
    for line in f:
      gold.append(line.rstrip())
  with open(google_loc) as f:
    for line in f:
      google.append(line.rstrip())
  return (gold, google)

def read_queries(fname):
    queries = []
    with open(fname) as f:
        for line in f:
            queries.append(line.rstrip())
    return queries

def edit_distance(s1, s2):
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
    return m[len(s1)][len(s2)]

def edit_distance_detail(s1, s2):
    s1 = '~' + s1
    s2 = '~' + s2
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
            result.append((0,(s1[index1-1],s2[index2-1])))
            index1 = index1 - 1
            index2 = index2 - 1
        elif m[index1][index2] == m[index1][index2-1] + 1:
            result.append((1,(s1[index1-1],s2[index2-1])))
            index2 = index2 - 1
        elif index1 > 1 and m[index1][index2] == m[index1-1][index2] + 1:
            result.append((2, (s1[index1-2],s1[index1-1])))
            index1 = index1 - 1
        elif index1 > 1 and index2 > 1 and m[index1][index2] == m[index1-2][index2-2] + k:
            result.append((3, (s1[index1-2],s1[index1-1])))
            index1 = index1 - 2
            index2 = index2 - 2
    return result

def check_within_ed(candidate, query):
    total_ed = 0
    for i in range(len(candidate)):
        ed = edit_distance(candidate[i], query[i])
        total_ed += ed
    return total_ed

#def StupidBackoffProb_Q (candidate):
#    score = 0.0
#    aux1 = ''
#    aux2 = ''
#    for token in candidate:
#        if aux1 != '':
#            if aux2 != '':
#                if (aux2,aux1,token) in trigram_dict:
#                    score += log(trigram_dict[(aux2,aux1,token)])
#                    score -= log(bygram_dict[(aux2,aux1)])
#                elif (aux1,token) in bygram_dict:
#                    score += log(0.4 * bygram_dict[(aux1,token)])
#                    score -= log(unigram_dict[aux1])
#                elif token in unigram_dict:
#                    score += log(unigram_dict[token])
#                    score -= log(total_term_num)
#                else:
#                    score = float('-inf')
#            aux2 = aux1
#        aux1 = token
#    return score

def calculateProb_Q (candidate):
    prob_Q = log(float(unigram_dict[candidate[0]])/total_term_num)
    if (len(candidate) > 1):
        for tup in itertools.izip(candidate[:-1], candidate[1:]):
            prob_mle = 0
            if tup in bygram_dict:
                prob_mle = float(bygram_dict[tup])/unigram_dict[tup[0]]
            prob_int = lam * unigram_dict[tup[1]] / total_term_num + (1 - lam) * prob_mle
            prob_Q += log(prob_int)
    return prob_Q

def process_candidate(candidate, query):
    if (not allUniExists(candidate)): return float("-inf")
    candidate = list(candidate)
    tmp_candidate = ' '.join(candidate)
    total_ed = edit_distance(tmp_candidate, query)      
    prob_Q = calculateProb_Q(candidate)
    prob_RQ = 0
    if(total_ed == 0): prob_RQ = log(prob_equal)
    else: prob_RQ = log(pow(prob_edit, total_ed)) + log(pow(1 - prob_edit, len(tmp_candidate) - total_ed))

    return prob_Q + prob_RQ

def process_query(query, model_type):
    query = joint_split_candidates(query).strip()
    candidates = generateCandidate(query)
    max_prob = float("-inf")
    max_candidate = ''
    count = 0
    for candidate in candidates:
        cur_prob = 0
        if model_type is 1:
            cur_prob = process_candidate(candidate, query)
        if model_type is 2:
            cur_prob = process_candidate_em(candidate, query)
        if cur_prob == float("-inf"): count += 1
        if (cur_prob > max_prob):
            max_prob = cur_prob
            max_candidate = candidate
    return ' '.join(max_candidate)

def process_queries(queries, model_type):
    correction = []
    for query in queries:
        tmp = process_query(query, model_type)
        if(len(tmp) == 0): tmp = query
        print >> sys.stdout, tmp
        correction.append(tmp)
    return correction

def cal_total_num():
    global total_term_num
    total_term_num = sum(unigram_dict.values())

def correct_rate(l1, l2):
    count = 0
    wrong_split = 0;
    for i in range(len(l1)):
        list_1 = l1[i].split()
        list_2 = l2[i].split()
        if len(list_1) is not len(list_2): wrong_split += 1
        if l1[i] == l2[i]:
            count += 1
    return (float(count) / len(l1), wrong_split)

def num_empty(l1):
    count = 0
    for one in l1:
        if (len(one) is 0):
                count += 1
    return count

def buildModel():
    global model_dict
    global uni_edit_dict
    global by_edit_dict
    global kgram_dict
    global bygram_dict
    global unigram_dict
    kgram_dict = unserialize_data('bygram_character')
    bygram_dict = unserialize_data('bygram_word')
    unigram_dict = unserialize_data('unigram')
    #    trigram_dict = unserialize_data('trigram_word')
    model_dict[0] = unserialize_data('sub.dict')
    model_dict[1] = unserialize_data('ins.dict')
    model_dict[2] = unserialize_data('del.dict')
    model_dict[3] = unserialize_data('trans.dict')
    uni_edit_dict = unserialize_data('uni_edit_dict')
    by_edit_dict = unserialize_data('by_edit_dict')

def get_uni_count(char):
    if char in uni_edit_dict: return uni_edit_dict[char]
    else: return 0

def get_by_count(tup):
    if tup in by_edit_dict: return by_edit_dict[tup]
    else: return 0

def compute_prob(type, tup):
    if (type == 0): return float(1)/(get_uni_count(tup[1]) + len(alphabet))
    if (type == 1): return float(1)/(get_uni_count(tup[0]) + len(alphabet))
    if (type == 2): return float(1)/(get_by_count(tup[0]+tup[1]) + len(alphabet) * len(alphabet))
    if (type == 3): return float(1)/(get_by_count(tup[0]+tup[1]) + len(alphabet) * len(alphabet))

def process_candidate_em(candidate, query):
    if (not allUniExists(candidate)): return float("-inf")
    candidate = ' '.join(list(candidate))

    edit_list = edit_distance_detail(candidate, query)
    prob_RQ = 0
    if (len(edit_list) is 0): prob_RQ = log(prob_equal)
    for edit in edit_list:
        if(edit[1] not in model_dict[edit[0]]): prob_RQ += log(compute_prob(edit[0], edit[1]))
        else: prob_RQ += log(model_dict[edit[0]][edit[1]])
    
    candidate = candidate.split()
    prob_Q = calculateProb_Q(candidate)
    return prob_Q + prob_RQ


if __name__ == '__main__':
    model_type = 0
    if (sys.argv[1] == 'uniform'): model_type = 1
    else: model_type = 2
    buildModel()
    cal_total_num()
    find_valid_alphabet()
    queries = read_queries(sys.argv[2])
#    (gold, google) = read_query_data()
    correction = process_queries(queries, model_type)
#    print 'correction rate: ' + `correct_rate(gold, correction)`
