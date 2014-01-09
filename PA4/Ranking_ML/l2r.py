#!/usr/bin/env python

### Module imports ###
import sys
import math
import re
import numpy as np
from sklearn import linear_model, svm, preprocessing
from math import log
from collections import OrderedDict
from numpy import dot
from numpy import array
import marshal
from itertools import combinations

avlen_body = 0;
avlen_url = 0;
avlen_title = 0;
avlen_header = 0;
avlen_anchor = 0;
cu = 0.4
ct = 0.2
cb = 0.05
ch = 0.05
ca = 0.3
B = 1.9
B_url = 0.6
B_title = 0.8
B_body = 0.8
B_header = 0.8
B_anchor = 0.6
lam = 1
lam1 = 0
K1 = 1.2
doc_vec_smooth = 500
score_thre = 1.0
idf_dict = {}

#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair
def extractFeatures(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    features = {}
    global avlen_body;
    global avlen_url;
    global avlen_title;
    global avlen_header;
    global avlen_anchor;
    url_count = 0;
    header_count = 0;
    anchor_count = 0;
    body_length = 0;
    url_length = 0;
    title_length = 0;
    header_length = 0;
    anchor_length = 0;
    for line in f:
        key = line.split(':', 1)[0].strip()
        value = line.split(':', 1)[-1].strip()
        if(key == 'query'):
            query = value
            queries[query] = []
            features[query] = {}
        elif(key == 'url'):
            url_count += 1
            url = value
            url_length_cur = len(re.split('//|/|\.|\?|\=',url.strip('/')))
            url_length += url_length_cur
            queries[query].append(url)
            features[query][url] = {}
            features[query][url]['url_length'] = url_length_cur
        elif(key == 'title'):
            title_length_cur = len(value.split())
            title_length += title_length_cur
            features[query][url][key] = value
            features[query][url]['title_length'] = title_length_cur
        elif(key == 'header'):
            header_count += 1
            curHeader = features[query][url].setdefault(key, [])
            curHeader.append(value)
            header_length_cur = len(value.split())
            header_length += header_length_cur
            features[query][url][key] = curHeader
            if ('header_length' not in features[query][url]):
                features[query][url]['header_length'] = header_length_cur
                features[query][url]['header_count'] = 1;
            else:
                features[query][url]['header_length'] += header_length_cur
                features[query][url]['header_count'] += 1;
        elif(key == 'body_hits'):
            if key not in features[query][url]:
                features[query][url][key] = {}
            temp = value.split(' ', 1)
            features[query][url][key][temp[0].strip()] \
                = [int(i) for i in temp[1].strip().split()]
        elif(key == 'body_length'):
            body_length += int(value)
            features[query][url][key] = int(value)
        elif (key == 'pagerank'):
            features[query][url][key] = int(value)
        elif(key == 'anchor_text'):
            anchor_text = value
            if 'anchors' not in features[query][url]:
                features[query][url]['anchors'] = {}
        elif(key == 'stanford_anchor_count'):
            anchor_count += int(value)
            features[query][url]['anchors'][anchor_text] = int(value)
            anchor_length += len(anchor_text.split())*int(value)
            if ('anchor_length' not in features[query][url]):
                features[query][url]['anchor_length'] = len(anchor_text.split())*int(value)
                features[query][url]['anchor_count'] = int(value)
            else:
                features[query][url]['anchor_length'] += len(anchor_text.split())*int(value)
                features[query][url]['anchor_count'] += int(value)
    
    avlen_anchor = float(anchor_length)/anchor_count
    avlen_body = float(body_length)/url_count
    avlen_title = float(title_length)/url_count
    avlen_header = float(header_length)/header_count
    avlen_url = float(url_length)/url_count
    f.close()
    return (queries, features)

def extractRawScoreVectors(query, features, url):
    query = query.lower();
    words = query.strip().split();
    words = list(OrderedDict.fromkeys(words));
    feature = features[query][url];
    Vectors = {};
    Vectors['urlVec'] = [0] * len(words);
    Vectors['titleVec'] = [0] * len(words);
    Vectors['bodyVec'] = [0] * len(words);
    Vectors['headerVec'] = [0] * len(words);
    Vectors['anchorVec'] = [0] * len(words);
    for i in range(len(words)):
        words[i] = words[i].strip()
        Vectors['urlVec'][i] += url.lower().count(words[i])
        if 'title' in feature:
            Vectors['titleVec'][i] += feature['title'].lower().count(words[i])
        if 'header' in feature:
            for header in feature['header']:
                Vectors['headerVec'][i] += header.lower().count(words[i])
        if 'anchors' in feature:
            for anchor_text in feature['anchors']:
                Vectors['anchorVec'][i] += feature['anchors'][anchor_text] * anchor_text.lower().count(words[i]);
        if 'body_hits' in feature:
            if words[i] in feature['body_hits']:
                Vectors['bodyVec'][i] += len(feature['body_hits'][words[i]])
    return Vectors

def rsTotf(vec):
    for i in range(len(vec)):
        if vec[i] > 0:
            vec[i] = 1 + log(vec[i])

def sublinearScaling(Vectors):
    for key in Vectors:
        rsTotf(Vectors[key])
    return Vectors

def lenNormalization(body_length, vec):
    for i in range(len(vec)):
        vec[i] = float(vec[i])/(body_length + doc_vec_smooth)

def generateDocVecs(query, features, url):
    Vectors = extractRawScoreVectors(query, features, url)
    sublinearScaling(Vectors)
    body_length = features[query][url]['body_length']
    for key in Vectors:
        lenNormalization(body_length, Vectors[key])
    return Vectors

def unserialize_data(fname):
    with open(fname, 'rb') as f:
        return marshal.load(f)

def queryVec(query):
    words = query.strip().split();
    termFreq = []
    appeared = set()
    for i in range(len(words)):
        if words[i] not in appeared:
            termFreq.append(words.count(words[i]))
            appeared.add(words[i])
    rsTotf(termFreq)
    words = list(OrderedDict.fromkeys(words))
    idfVec = [0] * len(words)
    for i in range(len(words)):
        idfVec[i] = idf_dict[words[i]]
    vec = [a * b for a,b in zip(termFreq, idfVec)]
    return vec

#extract queries from the file
def getQueries(rankingFile):
    pat = re.compile('((^|\n)query.*?($|\n))')
    rankings = open(rankingFile,'r')
    res = filter(lambda x: not(x is '' or x=='\n'), pat.split(rankings.read()))
    
    for item in res:
        if (item.strip().startswith('query:')):
            query = item.strip()
        else:
            results = filter(lambda x: not(x=='' or x=='\n'),
                             re.findall('url: .*', item.strip()))
            yield(query, results)
    rankings.close()

#populate map with ground truth for each query
def getGroundTruthScores(groundTruthFile):
    groundTruth = {}
    for (query, results) in getQueries(groundTruthFile):
        groundTruth[query] = {}
        for res in results:
            temp = res.rsplit(' ', 1)
            url = temp[0].strip()
            rel = float(temp[1].strip())
            groundTruth[query][url] = rel
    return groundTruth

###############################
##### Point-wise approach #####
###############################
def pointwise_train_features(train_data_file, train_rel_file):
    X = []
    Y = []
    (queries, features) = extractFeatures(train_data_file)
    groundTruth = getGroundTruthScores(train_rel_file)
    for query in features:
        queVec = array(queryVec(query))
        for url in features[query]:
            Vectors = generateDocVecs(query, features, url)
            example = []
            for field in Vectors:
                tf = array(Vectors[field])
                score = dot(queVec, tf)
                example.append(score)
            X.append(example)
            Y.append(groundTruth['query: ' + query]['url: ' + url])
    return (X, Y)

def pointwise_test_features(test_data_file):
    X = []
    (queries, features) = extractFeatures(test_data_file)
    queries = []
    index_map = {}
    counter = 0;
    for query in features:
      queries.append(query)
      queVec = array(queryVec(query))
      index_map[query] = {}
      for url in features[query]:
          index_map[query][url] = counter
          Vectors = generateDocVecs(query, features, url)
          example = []
          for field in Vectors:
              tf = array(Vectors[field])
              score = dot(queVec, tf)
              example.append(score)
          X.append(example)
          counter+=1
    return (X, queries, index_map)

def pointwise_learning(X, y):
  model = linear_model.LinearRegression()
  model.fit(X,y)
  return model

def pointwise_testing(X, model):
  return model.predict(X)

##############################
##### Pair-wise approach #####
##############################
def pairwise_train_features(train_data_file, train_rel_file):
    X = []
    Y = []
    map = {}
    (queries, features) = extractFeatures(train_data_file)
    groundTruth = getGroundTruthScores(train_rel_file)
    q_index = 0;
    counter = 0;
    for query in features:
        map[q_index] = []
        queVec = array(queryVec(query))
        for url in features[query]:
            map[q_index].append(counter)
            counter += 1
            Vectors = generateDocVecs(query, features, url)
            example = []
            for field in Vectors:
                tf = array(Vectors[field])
                score = dot(queVec, tf)
                example.append(score)
            X.append(array(example))
            Y.append(groundTruth['query: ' + query]['url: ' + url])
        q_index += 1

    X = preprocessing.scale(X);
    X_train = []
    Y_train = []
    for i in map:
        for(m,n) in combinations(map[i], 2):
            if(Y[m] > Y[n]):
                X_train.append(X[m]-X[n])
                Y_train.append(1)
            elif(Y[m] < Y[n]):
                X_train.append(X[m]-X[n])
                Y_train.append(-1)
    return (X_train, Y_train)

def pairwise_test_features(test_data_file):
    X = []
    (queries, features) = extractFeatures(test_data_file)
    queries = []
    index_map = {}
    counter = 0;
    for query in features:
        queries.append(query)
        queVec = array(queryVec(query))
        index_map[query] = {}
        for url in features[query]:
            index_map[query][url] = counter
            Vectors = generateDocVecs(query, features, url)
            example = []
            for field in Vectors:
                tf = array(Vectors[field])
                score = dot(queVec, tf)
                example.append(score)
            X.append(array(example))
            counter+=1
    return (X, queries, index_map)


def pairwise_learning(X, y):
    model = svm.SVC(kernel='linear', C=1.0)
    model.fit(X,y)
    return model

def pairwise_testing(X, model):
    return model.predict(X)

#################################################
##### Pair-wise approach with more features #####
#################################################
def endsWithPDF (url):
    if (url[len(url)-4:] == '.pdf'):
        result = 1
    else:
        result = -1
    return result

def containsWWW (url):
    if ('www' in url):
        result = 1
    else:
        result = -1
    return result

def generateStructVec(url,query,features):
    result = [0,0]
    if ('header' in features[query][url]):
        result[0] = 1
    if ('anchors' in features[query][url]):
        result[1] = 1
    return result


#inparams
#  query: the query that is processing
#  features: map containing features for each query,url pair
#  url: the url that is processing
#return value
#  Vectors: the vector containing the score of each query term for each field
def extractScoreVectors(query, features, url):
    query = query.lower();
    words = query.strip().split();
    words = list(OrderedDict.fromkeys(words));
    feature = features[query][url];
    Vectors = {};
    Vectors['urlVec'] = [0] * len(words);
    Vectors['titleVec'] = [0] * len(words);
    Vectors['bodyVec'] = [0] * len(words);
    Vectors['headerVec'] = [0] * len(words);
    Vectors['anchorVec'] = [0] * len(words);
    for i in range(len(words)):
        words[i] = words[i].strip()
        Vectors['urlVec'][i] += url.lower().count(words[i])
        Vectors['urlVec'][i] /= (1+B_url*(float(feature['url_length'])/avlen_url-1))
        Vectors['titleVec'][i] += feature['title'].lower().count(words[i])
        Vectors['titleVec'][i] /= (1+B_title*(float(feature['title_length'])/avlen_title-1))
        if 'header' in feature:
            for header in feature['header']:
                Vectors['headerVec'][i] += header.lower().count(words[i])
            Vectors['headerVec'][i] /= (1+B_header*(float(feature['header_length'])/(avlen_header*feature['header_count'])-1))
        if 'anchors' in feature:
            for anchor_text in feature['anchors']:
                Vectors['anchorVec'][i] += feature['anchors'][anchor_text] * anchor_text.lower().count(words[i]);
            Vectors['anchorVec'][i] /= (1+B_anchor*(float(feature['anchor_length'])/(avlen_anchor*feature['anchor_count'])-1))
        if 'body_hits' in feature:
            if words[i] in feature['body_hits']:
                Vectors['bodyVec'][i] += len(feature['body_hits'][words[i]])
        Vectors['bodyVec'][i] /= (1+B_body*(float(feature['body_length'])/avlen_body-1))
    return Vectors

def getQuerieURLScore(query,url,features):
    query = query.lower();
    words = query.strip().split();
    words = list(OrderedDict.fromkeys(words));
    score = 0;
    DocVec = extractScoreVectors(query, features, url)
    tf_url = array(DocVec['urlVec'])
    tf_title = array(DocVec['titleVec'])
    tf_body = array(DocVec['bodyVec'])
    tf_header = array(DocVec['headerVec'])
    tf_anchor = array(DocVec['anchorVec'])
    for i in range(len(words)):
        words[i] = words[i].strip()
        w_d_t = cu * tf_url[i] + ct * tf_title[i] + cb * tf_body[i] + ch * tf_header[i] + ca * tf_anchor[i]
        score += float(w_d_t)/(K1+w_d_t)*idf_dict[words[i]]
    pagerank_cur = features[query][url]['pagerank']
    if  pagerank_cur != 0:
        score += lam*log(pagerank_cur+lam1)
    return score

def smallestWindow(query, sentence):
    qs = query.strip().split();
    qs = list(OrderedDict.fromkeys(qs));
    sentence = sentence.lower()
    words = sentence.strip().split();
    smallest = float('inf')
    (s_start, s_end) = (0, 0)
    for start in range(len(words) - len(qs) + 1):
        remain = set(qs)
        for i in range(start, len(words)):
            if words[i] in remain:
                remain.remove(words[i])
            if len(remain) == 0:
                window = i - start + 1;
                if window < smallest:
                    smallest = window
                    (s_start, s_end) = (start, i)
                break
    return smallest

def getPositions(indices, hits):
    pos = []
    for i in range(len(indices)):
        pos.append(hits[i][indices[i]])
    return pos

def smallestWindowBody(query, body_hits):
    qs = query.strip().split();
    qs = list(OrderedDict.fromkeys(qs));
    smallest = float('inf')
    
    if (len(body_hits) < len(qs)): return smallest
    else:
        hits = []
        for word in body_hits:
            hits.append(body_hits[word])
        indices = [0] * len(hits)
        while True:
            curPos = getPositions(indices, hits)
            window = max(curPos) - min(curPos) + 1
            if window < smallest: smallest = window
            ind = curPos.index(min(curPos))
            if (indices[ind] == len(hits[ind]) - 1): break
            else: indices[ind] += 1
        return smallest

# smallest window vector to add
def smVec(query, features, url):
    query = query.lower();
    qLen = len(list(OrderedDict.fromkeys(query.strip().split())))
    vec = [] # url, title, body, header, anchor in order
    feature = features[query][url]
    
    url_words = url.replace(':', ' ').replace('/',' ').replace('.',' ').replace('?',' ').replace('~', ' ').replace('=', ' ').replace('-', ' ').replace('%20', ' ').replace('%', ' ').replace('_',' ').replace('&',' ').split()
    for i in range(len(url_words)):
        url_words[i] = url_words[i].strip()
        url_words[i] = url_words[i].lower()
    sentence = ' '.join(url_words)
    url_smallest = smallestWindow(query, sentence)
    vec.append(url_smallest)
    
    if 'title' in feature:
        vec.append(smallestWindow(query, feature['title']))
    else: vec.append(float('inf'))
    
    if 'body_hits' in feature:
        vec.append(smallestWindowBody(query, feature['body_hits']))
    else: vec.append(float('inf'))
    
    smallest = float('inf')
    if 'header' in feature:
        for header in feature['header']:
            window = smallestWindow(query, header)
            if window < smallest: smallest = window
        vec.append(smallest)
    else: vec.append(float('inf'))
    
    smallest = float('inf')
    if 'anchors' in feature:
        for anchor_text in feature['anchors']:
            window = smallestWindow(query, anchor_text)
            if window < smallest: smallest = window
        vec.append(smallest)
    else: vec.append(float('inf'))
    for i in range(len(vec)):
        vec[i] = 1 + B * float(1)/(1 + vec[i] - qLen)
    return vec

def numOfWindow(words, sentence):
    sentence = sentence.lower()
    sen = sentence.strip().split();
    counter = 0
    remain = set(words)
    
    for i in range(len(sen)):
        if sen[i] in remain: remain.remove(sen[i])
        if len(remain) == 0:
            counter += 1
            remain = set(words)
    
    return counter

def topWords(query):
    query = query.lower()
    vec = queryVec(query)
    qs = query.strip().split();
    qs = list(OrderedDict.fromkeys(qs));
    words_Score = {}
    for i in range(len(qs)):
        words_Score[qs[i]] = vec[i]
    
    sorted_words = list(sorted(words_Score, key = lambda x: words_Score[x], reverse = True))
    length = int(len(sorted_words) * score_thre)
    if length == 0: length = 1
    return sorted_words[: length]

def getBodyHits(query, url, words, features):
    hits = {}
    if 'body_hits' in features[query][url]:
        body_hits = features[query][url]['body_hits']
        for word in words:
            if word in body_hits:
                hits[word] = body_hits[word]
    return hits

def numWindowBody(query, url, qs, features):
    body_hits = getBodyHits(query, url, qs, features)
    if (len(body_hits) < len(qs)): return 0
    min_Len = float('inf')
    for word in body_hits:
        if(len(body_hits[word]) < min_Len): min_Len = len(body_hits[word])
    return min_Len

def nmVec(query, features, url):
    query = query.lower();
    topwords = topWords(query)
    #    shortQuery = ' '.join(topwords)
    
    vec = [] # url, title, body, header, anchor in order
    feature = features[query][url]
    
    url_words = url.replace(':', ' ').replace('/',' ').replace('.',' ').replace('?',' ').replace('~', ' ').replace('=', ' ').replace('-', ' ').replace('%20', ' ').replace('%', ' ').replace('_',' ').replace('&',' ').split()
    for i in range(len(url_words)):
        url_words[i] = url_words[i].strip()
        url_words[i] = url_words[i].lower()
    sentence = ' '.join(url_words)
    url_num = numOfWindow(topwords, sentence)
    vec.append(url_num)
    
    if 'title' in feature:
        vec.append(numOfWindow(topwords, feature['title']))
    else: vec.append(0)
    
    if 'body_hits' in feature:
        vec.append(numWindowBody(query, url, topwords, features))
    else: vec.append(0)
    
    nw = 0
    if 'header' in feature:
        for header in feature['header']:
            nw += numOfWindow(topwords, header)
        vec.append(nw)
    else: vec.append(0)
    
    nw = 0
    if 'anchors' in feature:
        for anchor_text in feature['anchors']:
            nw += numOfWindow(topwords, anchor_text) * feature['anchors'][anchor_text]
        vec.append(nw)
    else: vec.append(0)
    return vec

def pairwise_train_features_more(train_data_file, train_rel_file):
    X = []
    Y = []
    map = {}
    (queries, features) = extractFeatures(train_data_file)
    groundTruth = getGroundTruthScores(train_rel_file)
    q_index = 0;
    counter = 0;
    for query in features:
        map[q_index] = []
        queVec = array(queryVec(query))
        for url in features[query]:
            map[q_index].append(counter)
            counter += 1
            Vectors = generateDocVecs(query, features, url)
            example = []
            words = query.strip().split();
            words = list(OrderedDict.fromkeys(words));
            for field in Vectors:
                tf = array(Vectors[field])
                score = dot(queVec, tf)
                example.append(score)
            #example.append(endsWithPDF(url))
            example.append(getQuerieURLScore(query,url,features))
            swvec = smVec(query,features,url)
            example.append(swvec[2])
            example.append(features[query][url]['pagerank'])
            example.append(float(len(re.split('//|/|\.|\?|\=',url.strip('/'))))/len(words))
            example.extend(generateStructVec(url,query,features))
            example.append(containsWWW(url))
            X.append(array(example))
            Y.append(groundTruth['query: ' + query]['url: ' + url])
        q_index += 1
    
    X = preprocessing.scale(X);
    X_train = []
    Y_train = []
    for i in map:
        for(m,n) in combinations(map[i], 2):
            if(Y[m] > Y[n]):
                X_train.append(X[m]-X[n])
                Y_train.append(1)
            elif(Y[m] < Y[n]):
                X_train.append(X[m]-X[n])
                Y_train.append(-1)
    return (X_train, Y_train)

def pairwise_test_features_more(test_data_file):
    X = []
    (queries, features) = extractFeatures(test_data_file)
    queries = []
    index_map = {}
    counter = 0;
    for query in features:
        queries.append(query)
        queVec = array(queryVec(query))
        index_map[query] = {}
        for url in features[query]:
            index_map[query][url] = counter
            Vectors = generateDocVecs(query, features, url)
            words = query.strip().split();
            words = list(OrderedDict.fromkeys(words));
            example = []
            for field in Vectors:
                tf = array(Vectors[field])
                score = dot(queVec, tf)
                example.append(score)
            #example.append(endsWithPDF(url))
            example.append(getQuerieURLScore(query,url,features))
            swvec = smVec(query,features,url)
            example.append(swvec[2])
            example.append(features[query][url]['pagerank'])
            example.append(float(len(re.split('//|/|\.|\?|\=',url.strip('/'))))/len(words))
            example.extend(generateStructVec(url,query,features))
            example.append(containsWWW(url))
            X.append(array(example))
            counter+=1
    return (X, queries, index_map)

####################
####### PRank ######
####################
def PRank_features(train_data_file, train_rel_file):
    X = []
    Y = []
    (queries, features) = extractFeatures(train_data_file)
    groundTruth = getGroundTruthScores(train_rel_file)
    for query in features:
        queVec = array(queryVec(query))
        for url in features[query]:
            Vectors = generateDocVecs(query, features, url)
            example = []
            for field in Vectors:
                tf = array(Vectors[field])
                score = dot(queVec, tf)
                example.append(score)
            X.append(example)
            Y.append(groundTruth['query: ' + query]['url: ' + url])
    
    groundTruthMap = {}
    groundTruthSet = set()
    for query in groundTruth:
        for url in groundTruth[query]:
            groundTruthSet.add(groundTruth[query][url])
    groundTruthList = sorted(list(groundTruthSet))
    for i in range(len(groundTruthList)):
        groundTruthMap[groundTruthList[i]] = i
    for i in range(len(Y)):
        Y[i] = groundTruthMap[Y[i]]
    
    return (X, Y, len(groundTruthList))

def PRank_test_features(test_data_file):
    return pointwise_test_features(test_data_file)

def getr(w, x, b):
    for r in range(len(b)):
        if (dot(w,x) - b[r]) < 0:
            return r

def PRank_training(X, Y, k):
    w = array([0] * len(X[0]))
    b = [0] * (k-1)
    b.append(float("inf"))
    b = array(b)
    for m in range(10):
        for i in range(len(Y)):
            yt = getr(w,array(X[i]),b)
            if yt != Y[i]:
                yrt = []
                tau = []
                for j in range(k-1):
                    if Y[i] <= j: yrt.append(-1)
                    else: yrt.append(1)
                for l in range(k-1):
                    if((dot(w, array(X[i])) - b[l]) * yrt[l]) <= 0: tau.append(yrt[l])
                    else: tau.append(0)
                tau.append(0)
                yrt = array(yrt)
                tau = array(tau)
                w = w + sum(tau) * array(X[i])
                b = b - tau
    return (w, b)

def PRank_predict(X, w, b):
    Y = []
    for i in range(len(X)):
        Y.append(getr(w, X[i], b))
    return Y


####################
##### Training #####
####################
def train(train_data_file, train_rel_file, task):
  sys.stderr.write('\n## Training with feature_file = %s, rel_file = %s ... \n' % (train_data_file, train_rel_file))
  
  if task == 1:
    # Step (1): construct your feature and label arrays here
    (X, y) = pointwise_train_features(train_data_file, train_rel_file)
    
    # Step (2): implement your learning algorithm here
    model = pointwise_learning(X, y)
  elif task == 2:
    # Step (1): construct your feature and label arrays here
    (X, y) = pairwise_train_features(train_data_file, train_rel_file)
    
    # Step (2): implement your learning algorithm here
    model = pairwise_learning(X, y)
  elif task == 3: 
    # Add more features
    print >> sys.stderr, "Task 3\n"
    # Step (1): construct your feature and label arrays here
    (X, y) = pairwise_train_features_more(train_data_file, train_rel_file)
      
    # Step (2): implement your learning algorithm here
    model = pairwise_learning(X, y)
    

  elif task == 4: 
    # Extra credit
    # training part in the main function
    print >> sys.stderr, "Extra Credit\n"

  else: 
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 2]
    model = linear_model.LinearRegression()
    model.fit(X, y)
  
  # some debug output
  weights = model.coef_
  print >> sys.stderr, "Weights:", str(weights)

  return model 

def sortTest(index_map, y):
    rankedQueries = {}
    for query in index_map:
        scores = {}
        for url in index_map[query]:
            scores[url] = y[index_map[query][url]]
        rankedQueries[query] = list(sorted(scores, key = lambda x: scores[x], reverse = True))
    return rankedQueries

def printRankedResults(queries, outputFile):
    file = open(outputFile, 'w')
    for query in queries:
        print >> sys.stdout, ("query: " + query)
        file.write("query: " + query + "\n")
        for res in queries[query]:
            print >> sys.stdout, ("  url: " + res)
            file.write("  url: " + res + "\n")

###################
##### Testing #####
###################
def test(test_data_file, model, task):
  sys.stderr.write('\n## Testing with feature_file = %s ... \n' % (test_data_file))

  if task == 1:
    # Step (1): construct your test feature arrays here
    (X, queries, index_map) = pointwise_test_features(test_data_file)
    
    # Step (2): implement your prediction code here
    y = pointwise_testing(X, model)
    rankedQueries = sortTest(index_map, y)
    printRankedResults(rankedQueries, 'ranked.txt')
    
  elif task == 2:
    # Step (1): construct your test feature arrays here
    (X, queries, index_map) = pairwise_test_features(test_data_file)    
    # Step (2): implement your prediction code here
    y = dot(X, array(model.coef_[0]))
    rankedQueries = sortTest(index_map, y)
    printRankedResults(rankedQueries, 'ranked.txt')
    
  elif task == 3:
    # Add more features
    print >> sys.stderr, "Task 3\n"
    # Step (1): construct your test feature arrays here
    (X, queries, index_map) = pairwise_test_features_more(test_data_file)
    # Step (2): implement your prediction code here
    y = dot(X, array(model.coef_[0]))
    rankedQueries = sortTest(index_map, y)
    printRankedResults(rankedQueries, 'ranked.txt')
    

  elif task == 4: 
    # Extra credit
    # Testing part in the main function
    print >> sys.stderr, "Extra credit\n"

  else:
    queries = ['query1', 'query2']
    index_map = {'query1' : {'url1':0}, 'query2': {'url2':1}}
    X = [[0.5, 0.5], [1.5, 1.5]]  
    y = model.predict(X)
  
  # some debug output
#  for query in queries:
#    for url in index_map[query]:
#      print >> sys.stderr, "Query:", query, ", url:", url, ", value:", y[index_map[query][url]]

if __name__ == '__main__':
  sys.stderr.write('# Input arguments: %s\n' % str(sys.argv))
  idf_dict = unserialize_data('idf_dict')
  if len(sys.argv) != 5:
    print >> sys.stderr, "Usage:", sys.argv[0], "train_data_file train_rel_file test_data_file task"
    sys.exit(1)
  train_data_file = sys.argv[1]
  train_rel_file = sys.argv[2]
  test_data_file = sys.argv[3]
  task = int(sys.argv[4])
  print >> sys.stderr, "### Running task", task, "..."
# Extra credit training and testing:
  if task is 4:
    (X, Y, k) = PRank_features(train_data_file, train_rel_file)
    (w, b) =  PRank_training(X, Y, k)
    (X, queries, index_map) = PRank_test_features(test_data_file)
    y = dot(X, w)
    rankedQueries = sortTest(index_map, y)
    printRankedResults(rankedQueries, 'ranked.txt')
  else:
    model = train(train_data_file, train_rel_file, task)
    test(test_data_file, model, task)
