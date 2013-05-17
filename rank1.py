import sys
import re
from math import log
import math
from collections import Counter
import marshal

c_url = .3
c_body = .8
c_title = 3.7
c_anchor = 11
c_header = 1.8
smoothing_factor = 500
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

def scale(raw):
  return raw
  # if raw == 0:
  #   return 0.0
  # return float(1)+math.log(raw)

def get_idf(df):
  # check N
  return math.log((float(98998)+1)/(df+1))

def cosine_score(features, url, query):
  # fetching doc info with original query string
  doc_info = features[query][url]

  # removes query term duplicates
  query_with_dupes = query.split()
  query = list(set(query.split()))
  query_vector = []

  # weight query tf by idf
  for qw in query:
    query_vector.append(1.0 * get_idf(df_dict[qw]))
    #query_vector.append(query_with_dupes.count(qw) * get_idf(df_dict[qw]))

  # make doc vector
  url_vector = []
  title_vector = []
  body_vector = []
  header_vector = []
  anchor_vector = []
  body_length = doc_info['body_length']
  normalization_factor = body_length + smoothing_factor
  
  #url
  tokens = re.findall(r"\w+",url.lower())
  for w in query:
    if w in tokens:
      url_vector.append(c_url * scale(tokens.count(w)) / normalization_factor)
    else:
      url_vector.append(0.0)
 
  #title
  if 'title' in doc_info:
    title = doc_info['title'].lower().split()
    for w in query:
      if w in title:
        title_vector.append(c_title * scale(title.count(w))/ normalization_factor)
      else:
        title_vector.append(0.0)
  else:
    title = [0.0] * len(query_vector)

  #body
  if 'body_hits' in doc_info:
    body_hits = doc_info['body_hits']
    for w in query:
      if w in body_hits.keys():
        body_vector.append(c_body * scale(len(body_hits[w]))/ normalization_factor)
      else:
        body_vector.append(0.0)
  else:
    body_vector = [0.0] * len(query_vector)

  #header
  if 'header' in doc_info:
    curheaders = doc_info['header']
    header_tokens = [word.lower() for header in curheaders for word in header.split()]
    for w in query:
      if w in header_tokens:
        header_vector.append(c_header * scale(header_tokens.count(w)) / normalization_factor)
      else:
        header_vector.append(0.0)
  else:
    header_vector = [0.0] * len(query_vector)

  #anchor fields
  if 'anchors' in doc_info:
    anchors = doc_info['anchors']
    for w in query:
      count = 0.0
      for anchor in anchors:
        if w in anchor.split():
          count += (anchors[anchor] * anchor.split().count(w))
      anchor_vector.append(c_anchor * scale(count) / normalization_factor)
  else:
    anchor_vector = [0.0] * len(query_vector)

  doc_vector = [url_vector[i] + title_vector[i] + body_vector[i] + header_vector[i] + anchor_vector[i] for i in range(len(query_vector))]
  score = sum([query_vector[i] * doc_vector[i] for i in range(len(query_vector))]) 
  return score

#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query
def baseline(queries, features):
    rankedQueries = {}
    for query in queries.keys():
      results = queries[query]
      rankedQueries[query] = sorted(results, key = lambda x: cosine_score(features, x, query) , reverse = True )

      #features[query][x].setdefault('body_hits', {}).values() returns the list of body_hits for all query terms
      #present in the document, empty if nothing is there. We sum over the length of the body_hits array for all
      #query terms and sort results in decreasing order of this number
      # rankedQueries[query] = sorted(results, 
      #                               key = lambda x: sum([len(i) for i in 
      #                               features[query][x].setdefault('body_hits', {}).values()]), reverse = True)

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
    (queries, features) = extractFeatures(featureFile)

    #calling baseline ranking system, replace with yours
    rankedQueries = baseline(queries, features)
    
    #print ranked results to file
    printRankedResults(rankedQueries)
       
if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments" 
    #get document frequencies from file
    df_dict = marshal.load(open('df.p', 'rb'))

    main(sys.argv[1])
