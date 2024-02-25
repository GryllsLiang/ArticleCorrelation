# -*- coding: utf-8 -*-

import os
import sys
import string
import pyspark
import itertools
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

data_path = sys.argv[1]
stopwords_path = sys.argv[2]
out_file_path = sys.argv[3]
# Initialize context
conf = pyspark.SparkConf()
sc = pyspark.SparkContext(conf=conf)
sqlContext = pyspark.SQLContext(sc)

# Step 0 ======================================================================
# Preprocessing:
# 1. Get needed data and lowercase, remove stopwords, drop punctuation,
# lemmatize, drop independent numbers, remove stray spaces
# =============================================================================
stopwords = []

# read stopwords
with open(stopwords_path, 'r') as f:
    for line in f:
        stopwords.append(line.strip())
# Functions used for preprocessing
def remove_stopwords(text):
    return " ".join([x for x in text.split(" ") if x not in stopwords])

def strip_punctuation(text):
    return text.translate(str.maketrans(dict.fromkeys(string.punctuation)))

def remove_stray_spaces(text):
    return " ".join(text.split())

def is_indep_number(s):
    if s.isdigit():
        return True
    try:
        float(s)
        return True
    except:
        return False

def remove_indep_numbers(text):
    return " ".join([x for x in text.split(" ") if not is_indep_number(x)])

def filter_empty_and_none(text):
    return text is not None and len(text) > 0

def lem(text):
  lemmatizer = WordNetLemmatizer()
  lemmatized_tokens = []
  tokens = word_tokenize(text)
  pos_tags = pos_tag(tokens)
  # tag words
  for word, tag in pos_tags:
      if tag.startswith('J'):
          # adj.
          lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='a'))
      elif tag.startswith('V'):
          # verb.
          lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='v'))
      elif tag.startswith('N'):
          # noun.
          lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='n'))
      elif tag.startswith('R'):
          # adv.
          lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='r'))
      else:
          # others
          lemmatized_tokens.append(lemmatizer.lemmatize(word, pos='n'))

  return ' '.join(lemmatized_tokens)

nltk.download('popular')
datafile_rdd = sqlContext.read.json(data_path).rdd
title_rdd0 = (
    datafile_rdd
    .map(lambda x:x['title'])
    .map(lambda x: x.lower())
    .map(strip_punctuation)
    .map(remove_indep_numbers)
    .map(remove_stray_spaces)
    .filter(filter_empty_and_none)
)
title_rdd = (
    title_rdd0
    .map(remove_stopwords)
    .map(lem)
    .map(remove_indep_numbers)
    .map(remove_stray_spaces)
    .filter(filter_empty_and_none)
)
datafile_rdd = sqlContext.read.json(data_path).rdd

abstract_rdd0 = (
    datafile_rdd
    .map(lambda x:x['abstract'])
    .map(lambda x: x.lower())
    .map(strip_punctuation)
    .map(remove_indep_numbers)
    .map(remove_stray_spaces)
    .filter(filter_empty_and_none)
)
abstract_rdd = (
    abstract_rdd0
    .map(remove_stopwords)
    .map(lem)
    .map(remove_indep_numbers)
    .map(remove_stray_spaces)
    .filter(filter_empty_and_none)
)
# Step 1 ======================================================================
# Compute TF-IDF of every word w.r.t an abstract.
# Use key-value pair RDD and the groupByKey() or reduceByKey() API for this step.
# =============================================================================
# Tow functions for computing tf of abstract and title
def abstract_compute_tf(partition):
    # partition is Iterator<(word, 1)>
    for i in range(len(abstract_lines_collect)):
      count_dict = dict()
      for word in abstract_lines_collect[i]:
        if word in count_dict:
          count_dict[(i,word)] += 1
        else:
          count_dict[(i,word)] = 1
        # emit, use(iDoc,word) as key
        yield((i,word),1)

def title_compute_tf(partition):
    # partition is Iterator<(word, 1)>
    for i in range(len(title_lines_collect)):
      count_dict = dict()
      for word in title_lines_collect[i]:
        if word in count_dict:
          count_dict[(i,word)] += 1
        else:
          count_dict[(i,word)] = 1
        # emit, use(iDoc,word) as key
        yield((i,word),1)

# Get all words appear in abstract in one line
abstract_allwords = (
    abstract_rdd.flatMap(lambda x: x.split(" "))
)
# a matrix of words in each abstract
abstract_lines_collect = (
    abstract_rdd.map(lambda x: x.split(" "))
    .collect()
)

abstract_tf = (
    abstract_allwords.map(lambda x: (x, 1))
    .mapPartitions(abstract_compute_tf)
    .reduceByKey(lambda x, y: x+y)
)

# Get all words appear in title
title_allwords = (
    title_rdd.flatMap(lambda x: x.split(" "))
)
# a matrix of words in each abstract
title_lines_collect = (
    title_rdd.map(lambda x: x.split(" "))
    .collect()
)

title_tf = (
    title_allwords.map(lambda x: (x, 1))
    .mapPartitions(title_compute_tf)
    .reduceByKey(lambda x, y: x+y)
)

abstract_allwords_collect = abstract_allwords.collect()
abstract_tf_collect = abstract_tf.collect()

title_allwords_collect = title_allwords.collect()
title_tf_collect = title_tf.collect()

numDocs = len(abstract_lines_collect)

# Fuction for computing DF
def abstract_compute_df(partition):
    # record_dict to see whether the word has appeared in the doc
    record_dict = dict()
    for element in partition:
      if element in record_dict:
        continue
      for abstract in abstract_lines_collect:
        if element in abstract:
          # if the word doesn't appear before, add it in the dict, else, frequency + 1
          if element not in record_dict:
              record_dict[element] = 1
          else:
              record_dict[element] += 1
      if element not in record_dict:
        continue  
      yield (element, record_dict[element])
# compute df
abstract_df = dict(
    abstract_allwords.mapPartitions(abstract_compute_df)
    .reduceByKey(lambda x, y: x + y)
    .collect()
)

title_df = dict(
    title_allwords.mapPartitions(abstract_compute_df)
    .reduceByKey(lambda x, y: x + y)
    .collect()
)

# Step 3 ======================================================================
# Compute normalized TF-IDF of every word w.r.t. an abstract.
# If the TF-IDF value of word1 in doc1 is t􀬵 and the sum of squares of the TF-IDF
# of all the words in doc1 is S, then the normalized TF-IDF value of word1 is t1/sqrt(S).
# =============================================================================
def abstract_tfidf_partition(partition):
  #compute tfidf
    partition = list(partition)
    tfidf_dict = dict.fromkeys(abstract_df.keys(), 0)
    for element in partition:
        index, tf = element
        doc_num, word = index[0], index[1]
        if abstract_df[word] != 0:
          yield((doc_num,word), (1 + np.log10(tf)) * np.log10(numDocs / abstract_df[word]))

def title_tfidf_partition(partition):
    #compute tfidf
    partition = list(partition)
    tfidf_dict = dict.fromkeys(title_df.keys(), 0)
    for element in partition:
        index, tf = element
        doc_num, word = index[0], index[1]
        if word not in title_df:
          continue
        if title_df[word] != 0:
          yield((doc_num,word), (1 + np.log10(tf)) * np.log10(numDocs / title_df[word]))

abstract_tfidf = (
    abstract_tf.mapPartitions(abstract_tfidf_partition)
)

title_tfidf = (
    title_tf.mapPartitions(title_tfidf_partition)
)

def tfidf_normalized_partition(partition):
  #tfidf normalize
    partition = list(partition)
    S = sum(tfidf**2 for _, tfidf in partition)
    for element in partition:
        word, tfidf = element
        yield (word, tfidf / np.sqrt(S))

abstract_tfidf_normalized = dict(
    abstract_tfidf.mapPartitions(tfidf_normalized_partition)
    .groupByKey()
    .mapValues(list)
    .collect()
)

# Step 4 ======================================================================
# Take each title as a query, compute their normalized TF-IDF and compute the
# relevance of each abstract w.r.t a query. (When you compute the normalized TF-IDF of
# titles, TF is the term frequency in the title, but DF is the document frequency in the
# abstracts.)
# =============================================================================

title_tfidf_normalized = dict(
    title_tfidf.mapPartitions(tfidf_normalized_partition)
    .groupByKey()
    .mapValues(list)
    .collect()
)

def getMat(tfidf,allwords):
  #transform array into matrix with column names
  # tfidf is the array of array
  # allwords is used for building columns
  # return numDocs * numWords matrix, column names
  unique_words = set(allwords)

  num_unique_words = len(set(allwords))

  column_dict = dict()
  columns=[]
  # create column dict, connect every word with a No.
  for k,v in enumerate(unique_words):
    column_dict[v] = k
    columns.append(v)
  # create a zero matrix of the wanted shape, then fill the position with the value if it has
  normalized_tfidf_matrix = np.zeros((numDocs,num_unique_words))
  record_dict = dict()

  for k,v in abstract_tfidf_normalized.items():
    if k[1] not in column_dict:
      continue
    normalized_tfidf_matrix[k[0]][column_dict[k[1]]] = v[0]
    if (k[0],k[1]) not in record_dict:
      record_dict[(k[0],k[1])] = 0

  return pd.DataFrame(normalized_tfidf_matrix, columns=columns), columns

absract_normalized_tfidf_matrix, abstract_columns = getMat(abstract_tfidf_normalized, abstract_allwords_collect)
title_normalized_tfidf_matrix, title_columns = getMat(title_tfidf_normalized, title_allwords_collect)

absract_normalized_tfidf_matrix

title_normalized_tfidf_matrix

# generate query vector matrix
query_list = title_lines_collect
query_dict = [{i: 0 for i in abstract_columns} for _ in range(len(query_list))]
query_vectors = []
for i in range(len(query_list)):
  for word in query_list[i]:
    if word in query_dict[i]:
      query_dict[i][word] = 1
  query_vectors.append(list(query_dict[i].values()))
query_vectors = np.array(query_vectors)

title_list = (
    datafile_rdd
    .map(lambda x:x['title'])
    .collect()
)
abstract_list = (
    datafile_rdd
    .map(lambda x:x['abstract'])
    .collect()
)

title_list = title_rdd.collect()
abstract_list = abstract_rdd.collect()

# Step 5&6 ======================================================================
# For each query, sort and get the top-1 abstract.
# For each top-1 abstract, calculate whether it is a hit or miss. Calculate an accuracy
# score and print the accuracy score.
# =============================================================================
# Transverse list to matrix
topk = 3
# Compute similarity of each column of M with the q vector.
def cosine_similarity(q, M):
  return np.dot(M, q) / (np.linalg.norm(q) * np.linalg.norm(M))

count_hit = 0
count_top3 = 0
with open(out_file_path, 'w+') as f:
  for i in range(len(query_vectors)):
    similarity = cosine_similarity(query_vectors[i], absract_normalized_tfidf_matrix)
    relevances = sorted([(doc_id, relevance) for doc_id, relevance in enumerate(similarity)],
                    key=lambda x: x[1],
                    reverse=True)
    if relevances[0][0] == i:
      count_hit += 1
    else:
      f.write(f"Miss: query[{i}]: {title_list[i]}\nThe top {topk} relevances of the query are:\n")
      f.write(f"{relevances[0:topk]}\n")
      for iabstract in range(topk):
        f.write(f"Abstract[{relevances[iabstract][0]}]:\n")
        f.write(f"{abstract_list[relevances[iabstract][0]]}\n")
        # Coun top3 rate
        if relevances[iabstract][0] == i:
          count_top3 += 1
      f.write('=============================================================================\n')
  f.write(f'The hit rate is: {count_hit/numDocs}.\n')
  f.write(f'The top3 rate in all documents is: {(count_hit+count_top3)/numDocs}.\n')
  f.write(f'The top3 rate in miss documents is: {count_top3/(numDocs-count_hit)}.\n')

# Task 2
# Step 1 ======================================================================
# Compute the term frequency (TF) of every word (except for stopwords) in each
# abstract.
# =============================================================================
def getAbMat(Ablist, allwords):
  # transform array into matrix with column names
  # Ablist is the array of Abstract tf array
  # allwords is used for building columns
  # return numDocs * numWords matrix, column names
  unique_words = set(allwords)

  num_unique_words = len(set(allwords))

  column_dict = dict()
  columns=[]
  # create column dict, connect every word with a No.
  for k,v in enumerate(unique_words):
    column_dict[v] = k
    columns.append(v)
  # create a zero matrix of the wanted shape, then fill the position with the value if it has
  abstract_allwords_matrix = np.zeros((numDocs,num_unique_words))
  record_dict = dict()
  for k,v in Ablist:
    if k[1] not in column_dict:
      continue
    abstract_allwords_matrix[k[0]][column_dict[k[1]]] = v
    if (k[0],k[1]) not in record_dict:
      record_dict[(k[0],k[1])] = 0
  return pd.DataFrame(abstract_allwords_matrix, columns=columns),columns

abstract_tf_matrix,_ = getAbMat(abstract_tf_collect ,abstract_allwords_collect)
abstract_tf_matrix

# Step 2 ======================================================================
# For each category, sum the TF of words for all abstracts, producing a vector
# representation for each category.
# =============================================================================
category_list = (
    datafile_rdd
    .map(lambda x:x['categories'])
    .collect()
)
categories_abstract_rdd = sc.parallelize(zip(category_list, abstract_lines_collect))

def myreduce(x):
  count_dict = dict()
  for i in range(len(x)):
    if x[i][0][0] not in count_dict:
      count_dict[x[i][0][0]] = 1
    else:
      count_dict[x[i][0][0]] += 1
  return count_dict

categories_abstract_rdd = (
    categories_abstract_rdd
    .map(lambda x: (x[0],list(map(lambda x: (x.split(" "),1),x[1])))) # split words of the same category, give category as key
    .reduceByKey(lambda x,y: x + y) # use category as group attributes, concat the words of same category together
    # x[0] :('Astrophysics ',{'extend': 7,'masshalo': 1})
    # x[0][1]:{'extend': 7,'masshalo': 1,...}
    .map(lambda x: (x[0], myreduce(x[1])))
)
categories_abstract_collect = categories_abstract_rdd.collect()

# Step 3 ======================================================================
# For each pair of categories, calculate the similarity score by calculating the cosine
# similarity of the vectors produced from Step 2.
# =============================================================================
def getCatMat(Catlist,allwords):
  # transform array into matrix with column names
  # Catlist is the array of Category tf array
  # allwords is used for building columns
  # return numDocs * numWords matrix, column names
  unique_words = set(allwords)

  num_unique_words = len(unique_words)

  Ncategory = len(Catlist)
  column_dict = dict()
  columns=[]
  for k,v in enumerate(unique_words):
    column_dict[v] = k
    columns.append(v)
  cat_matrix = np.zeros((Ncategory,num_unique_words))
  record_dict = dict()
  for i in range(Ncategory):
    for k,v in Catlist[i][1].items():
      cat_matrix[i][column_dict[k]] = v

  return pd.DataFrame(cat_matrix, columns=columns), columns

categories_abstract_tf_matrix, category_columns = getCatMat(categories_abstract_collect, abstract_allwords_collect)

categories_abstract_tf_matrix

# calculate cosine_similarity of matrix
vec1 = np.expand_dims(categories_abstract_tf_matrix, axis=1)
vec2 = np.expand_dims(abstract_tf_matrix, axis=0)
dot_product = np.sum(vec1 * vec2, axis=2)
norm_product = np.linalg.norm(vec1, axis=2) * np.linalg.norm(vec2, axis=2)
cosine_similarity = dot_product / norm_product

category_columns = [i[0] for i in categories_abstract_collect]

# Step 4 ======================================================================
# Plot a heat map or other appropriate
# visualizations to show the correlation between categories. Save the plot as an image file.
# You may use the matplotlib, seaborn, or any other Python data visualization libraries.
# =============================================================================
# plot the heat map of category-abstract matrix
plt.figure(figsize=(20, 10))
sns.heatmap(cosine_similarity.transpose(),xticklabels=category_columns)

# to show the correlation among categories, plot correlation matrix heat map
cor_cosine_similarity = np.corrcoef(cosine_similarity)
plt.figure(figsize=(20, 10))
sns.heatmap(cor_cosine_similarity,xticklabels=category_columns,yticklabels=category_columns)

sc.stop()