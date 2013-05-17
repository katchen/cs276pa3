import sys
import re
from math import log
from __future__ import division

df_dict = {}

#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair
def extractFeatures(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    features = {}

    count_number = {'body': 0,
                    'url': 0,
                    'title': 0,
                    'header': 0,
                    'anchors': 0}
    count_total = {'body': 0,
                   'url': 0,
                   'title': 0,
                   'header': 0,
                   'anchors': 0}

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
        count_number[key] += 1
        count_total[key] += len(re.findall(r"\w+",url.lower()))
      elif(key == 'title'):
        features[query][url][key] = value
        count_number[key] += 1
        count_total[key] += len(value.split())
      elif(key == 'header'):
        curHeader = features[query][url].setdefault(key, [])
        curHeader.append(value)
        features[query][url][key] = curHeader
        count_number[key] += 1
        count_total[key] += len(value.split())
      elif(key == 'body_hits'):
        if key not in features[query][url]:
          features[query][url][key] = {}
        temp = value.split(' ', 1)
        features[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        features[query][url][key] = int(value)
        if (key == 'body_length'):
          count_number[key] += 1
          count_total[key] += int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        if 'anchors' not in features[query][url]:
          features[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        features[query][url]['anchors'][anchor_text] = int(value)
        count_number[key] += int(value)
        count_total[key] += int(value) * len(anchor_text.split())
      
    f.close()

    averages = {'body': count_total['body']/count_number['body'],
                'url': count_total['url']/count_number['url'],
                'title': count_total['title']/count_number['title'],
                'header': count_total['header']/count_number['header'],
                'anchors': count_total['anchors']/count_number['anchors']}


    return (queries, features, averages) 

def BM25F_score(query, features, averages, url):

  doc_info = features[query][url]
  q = list(set(query.split()))
  fields = ['body', 'url', 'title', 'header', 'anchors']
  W_f = {'body': 1,
         'url': 1,
         'title': 1,
         'header': 1,
         'anchors': 1}
  B_f = {'body': 1,
         'url': 1,
         'title': 1,
         'header': 1,
         'anchors': 1}

  for term in q:
    w_dt = 0
    for field in fields:
      tf = 0
      ftf = 0
      if field != 'url':
        tf = doc_info[field].count(term)
        ftf = tf / (1 + B_f[field] * \
          (len(doc_info[field].split()) / averages[field] - 1))
      else:
        tokens = re.findall(r"\w+",doc_info[field].lower())
        tf = tokens.count(term)
        ftf = tf / (1 + B_f[field] * \
          (len(tokens) / averages[field] - 1))
      w_dt += W_f[field] * ftf
    score = w_dt / (K1 + w_dt) * 



#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query
def baseline(queries, features, averages):
    rankedQueries = {}
    for query in queries.keys():
      results = queries[query]
      #features[query][x].setdefault('body_hits', {}).values() returns the list of body_hits for all query terms
      #present in the document, empty if nothing is there. We sum over the length of the body_hits array for all
      #query terms and sort results in decreasing order of this number
      rankedQueries[query] = sorted(results, 
                                    key = lambda x: BM25F_score(query, features, averages, x), reverse = True)

    return rankedQueries


#inparams
#  queries: contains ranked list of results for each query
#  outputFile: output file name
def printRankedResults(queries):
    for query in queries:
      print("query: " + query)
      for res in queries[query]:
        print("  url: " + res)

#inparams
#  featureFile: file containing query and url features
def main(featureFile):
    #output file name
    outputFile = "ranked.txt" #Please don't change this!

    #populate map with features from file
    (queries, features, averages) = extractFeatures(featureFile)

    #calling baseline ranking system, replace with yours
    rankedQueries = baseline(queries, features, averages)
    
    #print ranked results to file
    printRankedResults(rankedQueries)
       
if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments" 
    df_dict = marshal.load(open('df.p', 'rb'))
    main(sys.argv[1])
