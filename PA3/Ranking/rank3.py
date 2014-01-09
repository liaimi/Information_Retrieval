import sys
import re
from math import log
from collections import OrderedDict
from numpy import dot
from numpy import array
from operator import itemgetter
import marshal
import re

doc_vec_smooth = 500
cu_sw = 0.2
ct_sw = 0.2
cb_sw = 0.05
ch_sw = 0.05
ca_sw = 0.5
B = 1.9
idf_dict = {}

score_thre = 0.55

#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair
def extractFeatures(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    features = {}

    for line in f:
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if(key == 'query'):
        query = value
        queries[query] = []
        features[query] = {}
      elif(key == 'url'):
        url = value
        queries[query].append(url)
        features[query][url] = {}
      elif(key == 'title'):
        features[query][url][key] = value
      elif(key == 'header'):
        curHeader = features[query][url].setdefault(key, [])
        curHeader.append(value)
        features[query][url][key] = curHeader
      elif(key == 'body_hits'):
        if key not in features[query][url]:
          features[query][url][key] = {}
        temp = value.split(' ', 1)
        features[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        features[query][url][key] = int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        if 'anchors' not in features[query][url]:
          features[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        features[query][url]['anchors'][anchor_text] = int(value)
      
    f.close()
    return (queries, features) 

#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query
def baseline(queries, features):
    rankedQueries = {}
    for query in queries.keys():
      results = queries[query]
      #features[query][x].setdefault('body_hits', {}).values() returns the list of body_hits for all query terms
      #present in the document, empty if nothing is there. We sum over the length of the body_hits array for all
      #query terms and sort results in decreasing order of this number
      rankedQueries[query] = sorted(results, 
                                    key = lambda x: sum([len(i) for i in 
                                    features[query][x].setdefault('body_hits', {}).values()]), reverse = True)

    return rankedQueries

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

################################
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
    
#    rsTotf(Vectors['anchorVec'])
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
################################
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


def smVec(query, features, url):
    query = query.lower();
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
    return vec

def sortSingleQuery_sm(query, urls, features):
    scores = {}
    qLen = len(list(OrderedDict.fromkeys(query.strip().split())))
    
    for url in urls:
        swVec = smVec(query, features, url)
        
        for i in range(len(swVec)):
            swVec[i] = 1 + B * float(1)/(1 + swVec[i] - qLen)
        DocVec = generateDocVecs(query, features, url)
        tf_url = array(DocVec['urlVec'])
        tf_title = array(DocVec['titleVec'])
        tf_body = array(DocVec['bodyVec'])
        tf_header = array(DocVec['headerVec'])
        tf_anchor = array(DocVec['anchorVec'])
    
        QueVec = array(queryVec(query))
        score = dot(QueVec, (swVec[0] * cu_sw * tf_url + swVec[1] * ct_sw * tf_title + swVec[2] * cb_sw * tf_body + swVec[3] * ch_sw * tf_header + swVec[4] * ca_sw * tf_anchor))
        scores[url] = score
    #sorted_score =  sorted(scores.iteritems(), key = operator.itemgetter(1))
    return list(sorted(scores, key = lambda x: scores[x], reverse = True))

def smallestWindowMethod(queries, features):
    rankedQueries = {}
    for query in queries:
        rankedQueries[query] = sortSingleQuery_sm(query, queries[query], features)
    return rankedQueries
####################################
#inparams
#  queries: contains ranked list of results for each query
#  outputFile: output file name
def printRankedResults(queries, outputFile):
    file = open(outputFile, 'w')
    for query in queries:
      print >> sys.stdout, ("query: " + query)
      file.write("query: " + query + "\n")
      for res in queries[query]:
        print >> sys.stdout, ("  url: " + res)
        file.write("  url: " + res + "\n")

#inparams
#  featureFile: file containing query and url features
def main(featureFile):
    #output file name
    outputFile = "ranked.txt" #Please don't change this!

    #populate map with features from file
    (queries, features) = extractFeatures(featureFile)

    global idf_dict
    idf_dict = unserialize_data('idf_dict')
    rankedQueries = smallestWindowMethod(queries, features)
    #print ranked results to file
    printRankedResults(rankedQueries, outputFile)

if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments"
    main(sys.argv[1])
