import sys
import re
from math import log
import math
from collections import Counter
import marshal
import itertools

c_url = .3
c_body = .8
c_title = 3.7
c_anchor = 11
c_header = 1.8
smoothing_factor = 500
df_dict = {}
B = 2

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


def indexes_inbound(indexes, found):
  for qw in range(len(found)):
    if indexes[qw] >= len(found[qw]):
      return False
  return True

def get_window(indexes, found):
  values = get_values(indexes,found)
  return max(values) - min(values) + 1

def get_values(indexes,found):
  return [found[i][indexes[i]] for i in range(len(found))]

def get_smallest_window_from_map(query, found):
  # get first index
  indexes = [0] * len(found)
  sw = float('inf')
  while indexes_inbound(indexes, found):
    w = get_window(indexes, found)
    if w < sw:
      sw = w

    cur_min = min(get_values(indexes, found))
    for i in range(len(found)):
      if found[i][indexes[i]] == cur_min:
        indexes[i] = indexes[i]+1

  return sw

def get_smallest_window(features, url, query):
  doc_info = features[query][url]

  query = list(set(query.split()))

  url_has_window = True
  title_has_window = True
  body_has_window = True
  header_has_window = True
  anchor_has_window = True

  smallest_url_window = float('inf')
  smallest_title_window = float('inf')
  smallest_body_window = float('inf')
  smallest_header_window = float('inf')
  smallest_anchor_window = float('inf')

  found = []
  url_tokens = re.findall(r"\w+",url.lower())
  for qw in query:
    if qw in url_tokens:
      found.append([i for i in range(len(url_tokens)) if url_tokens[i] == qw])
    else:
      url_has_window = False
      break
  if url_has_window:
    smallest_url_window = get_smallest_window_from_map(query, found)

  found = []
  if 'title' in doc_info:
    title = doc_info['title'].lower().split()
    for qw in query:
      if qw in title:
        found.append([i for i in range(len(url_tokens)) if url_tokens[i] == qw])
      else:
        title_has_window = False
        break
    if title_has_window:
      smallest_title_window = get_smallest_window_from_map(query, found)
  
  found = []
  if 'body_hits' in doc_info:
    body_hits = doc_info['body_hits']
    if len(body_hits.keys()) != len(query):
      body_has_window = False
    else:
      for qw in query:
        found.append(body_hits[qw])
      smallest_body_window = get_smallest_window_from_map(query, found)

  found = []
  if 'header' in doc_info:
    curheaders = doc_info['header']
    for header in curheaders:
      header_bool = True
      found = []
      for qw in query:
        if qw in header.lower().split():
          found.append([i for i in range(len(url_tokens)) if url_tokens[i] == qw])
        else:
          header_bool = False
          break;
      if header_bool:
        temp = get_smallest_window_from_map(query, found)
        if temp < smallest_header_window:
          smallest_header_window = temp

  found = []
  if 'anchors' in doc_info:
    anchors = doc_info['anchors']
    for anchor in anchors:
      anchor_bool = True
      found = []
      for qw in query:
        if qw in anchor.lower().split():
          found.append([i for i in range(len(url_tokens)) if url_tokens[i] == qw])
        else:
          anchor_bool = False
          break;
      if anchor_bool:
        temp = get_smallest_window_from_map(query, found)
        if temp < smallest_anchor_window:
          smallest_anchor_window = temp

  return min([smallest_url_window, smallest_title_window, smallest_body_window, smallest_header_window, smallest_anchor_window])

def cosine_score(features, url, query):
  # fetching doc info with original query string
  doc_info = features[query][url]

  q = get_smallest_window(features, url, query)

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

  if q == float('inf'):
    boost = 1
  elif q == len(query):
    boost = B
  else:
    normal_q = q/len(query)
    boost = (B-1)* 1 / normal_q + 1

  doc_vector = [url_vector[i] + title_vector[i] + body_vector[i] + header_vector[i] + anchor_vector[i] for i in range(len(query_vector))]
  score = sum([query_vector[i] * doc_vector[i] for i in range(len(query_vector))]) 
  return score * boost

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
