import sys
import re
import math
from math import log
from collections import OrderedDict
from numpy import array
import marshal

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
B_url = 0.6
B_title = 0.8
B_body = 0.8
B_header = 0.8
B_anchor = 0.6
lam = 1
lam1 = 0
K1 = 1.2
idf_dict = {}

def unserialize_data(fname):
    with open(fname, 'rb') as f:
        return marshal.load(f)
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

#inparams
#  query: the query that is processing
#  features: map containing features for each query,url pair
#  urls: url lists for the query
#return value
#  ordered lists of urls for the query
def sortSingleQuery(query, urls, features):
    scores = {}
    query = query.lower();
    words = query.strip().split();
    words = list(OrderedDict.fromkeys(words));
    for url in urls:
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
        scores[url] = score
    return list(sorted(scores, key = lambda x: scores[x], reverse = True))

#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query
def BM25F(queries, features):
    rankedQueries = {}
    for query in queries:
        rankedQueries[query] = sortSingleQuery(query, queries[query], features)
    return rankedQueries

#inparams
#  queries: contains ranked list of results for each query
#  outputFile: output file name
def printRankedResults(queries, outputFile):
    file = open(outputFile, 'w')
    for query in queries:
      file.write("query: " + query + "\n")
      for res in queries[query]:
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
    
#    #calling BM25F ranking system
    rankedQueries = BM25F(queries, features)

    #print ranked results to file
    printRankedResults(rankedQueries, outputFile)

if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments" 
    main(sys.argv[1])

