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
cu = 0.3
ct = 0.2
cb = 0.05
ch = 0.05
ca = 0.4

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

def sortSingleQuery(query, urls, features):
    scores = {}
    for url in urls:
        DocVec = generateDocVecs(query, features, url)
        tf_url = array(DocVec['urlVec'])
        tf_title = array(DocVec['titleVec'])
        tf_body = array(DocVec['bodyVec'])
        tf_header = array(DocVec['headerVec'])
        tf_anchor = array(DocVec['anchorVec'])
        
        QueVec = array(queryVec(query))
        score = dot(QueVec, (cu * tf_url + ct * tf_title + cb * tf_body + ch * tf_header + ca * tf_anchor))
        scores[url] = score
#sorted_score =  sorted(scores.iteritems(), key = operator.itemgetter(1))
    return list(sorted(scores, key = lambda x: scores[x], reverse = True))
    

def cosSimilarity(queries, features):
    rankedQueries = {}
    for query in queries:
        rankedQueries[query] = sortSingleQuery(query, queries[query], features)    
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
    rankedQueries = cosSimilarity(queries, features)
    #print ranked results to file
    printRankedResults(rankedQueries, outputFile)

if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments"
    main(sys.argv[1])
