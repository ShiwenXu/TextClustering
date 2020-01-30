from __future__ import division
import collections
import operator
import nltk
import string
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer 
import math
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import statistics
import matplotlib.pylab as plt

# returns a list of dictionary: key is the documentID, value is a list of words in current document
def parseBody():
	# pase the corpus to document dictionary 

	# store every document in a list
	# each document is a dictionary {key is the newID, value is a list of word}
	list_of_all_docs = {}

	corpusFile = open ("/homes/xu931/project2/reut2-subset.sgm", 'r')
	soup = BeautifulSoup(corpusFile, 'html.parser')

	reuters = soup.findAll('reuters')

	bagofuniquewords = set()

	for every_doc in reuters:
		# save bag of words of current document 
		words_list = []
		# extract the document ID 
		docID = int(every_doc['newid'])
		# extract the body content 
		documentTag = every_doc.findAll('body')

		# if body cotent is NULL, bag of word should be empty list; 
		if not documentTag:
			list_of_all_docs[docID] = documentTag
		else:
			for body in documentTag:
				body_content = body.getText()
				body_content = str(body_content)
				body_content = body_content.translate(None, string.punctuation)

			words_list = nltk.word_tokenize(body_content)
			# remove last ending 'Reuter'
			words_list = words_list[:len(words_list)-2]
			words_list = [word.lower() for word in words_list]
			#append the value
			list_of_all_docs[docID] = words_list
			for word in words_list:
				bagofuniquewords.add(word)
	
	sorted_list_of_all_docs = sorted(list_of_all_docs.items(), key=operator.itemgetter(0), reverse=False)
	N = 0
	for key, value in sorted_list_of_all_docs:
		if len(value) != 0:
			N += 1 
	return sorted_list_of_all_docs, N, bagofuniquewords


def removeStopwords(sorted_list_of_all_docs, N, bagofuniquewords):
	df = {}
	initial_stopwords = nltk.corpus.stopwords.words('english')
	stopwords_list = []
	for word in initial_stopwords:
		stopwords_list.append(str(word))
	

	# count the frequency of a word in the corpus
	for word in bagofuniquewords:
		count = 0
		for key, value in sorted_list_of_all_docs:
			if word in value:
				count += 1
		df[word] = count 
	df = sorted(df.items(), key=operator.itemgetter(1), reverse=True)



	# get idf for each word
	idf = {}
	for word, count in df:
		if word not in idf:
			idf[word] = math.log(N/count)
		else:
			continue
	
	sorted_idf = sorted(idf.items(), key=operator.itemgetter(1), reverse=True)

	larget_idf = sorted_idf[0][1]
	smallest_idf = sorted_idf[-1][1]

	upperbound, lowerbound = float(larget_idf) * 0.95 , float(smallest_idf) * 1.05
	for word, value in idf.items():
		if value >= upperbound or value <= lowerbound:
				stopwords_list.append(word)
	
	# remove stopword in each document return new list_of_all_docs
	removedstopword_docs = {}

	for key, value in sorted_list_of_all_docs:
		if key not in removedstopword_docs:
			words_list = []
			for word in value:
				if word not in stopwords_list:
					words_list.append(word)			
			removedstopword_docs[key] = words_list
	sorted_removedstopword_docs = sorted(removedstopword_docs.items(), key=operator.itemgetter(0), reverse=False)
	return sorted_removedstopword_docs



def stemDoc(docs_without_stopwords):
	preprocessed_docs = {}
	ps = PorterStemmer()
	for key, value in docs_without_stopwords:
		if key not in preprocessed_docs:
			words_list = []
			for word in value:
				words_list.append(str(ps.stem(word)))
			preprocessed_docs[key] = words_list

	sorted_preprocess_docs = sorted(preprocessed_docs.items(), key=operator.itemgetter(0), reverse=False)
	return sorted_preprocess_docs



def getIDF(sorted_preprocessed_docs, N):
	bagwords = set()

	for key, value in sorted_preprocessed_docs:
		for word in value:
			bagwords.add(word)

	# count how many document contains such word
	df = {}
	for word in bagwords:
		count = 0
		for key, value in sorted_preprocessed_docs:
			if word in value:
				count += 1 
		df[word] = count
	df = sorted(df.items(), key=operator.itemgetter(1), reverse=True)

	# calculate the idf
	IDF = {}
	for word, count in df:
		if word not in IDF:
			IDF[word] = math.log(N/count)
		else:
			continue
	return IDF



def getTFIDF(sorted_preprocessed_docs, IDF):
	doc_with_tfidf_score = {}
	for key, value in sorted_preprocessed_docs:
		if key not in doc_with_tfidf_score:
			tfidf_each_word = {}
			for word in value:
				if word not in tfidf_each_word:
					# term frequency
					term_freq = math.log(value.count(word)) + 1 
					# find its according idf
					idf_score = IDF[word]
					tfidf_each_word[word] = term_freq * idf_score
			doc_with_tfidf_score[key] = tfidf_each_word
		# uncomment for all docs
		# break
	sorted_doc_with_tfidf_score = sorted(doc_with_tfidf_score.items(), key=operator.itemgetter(0), reverse=False)
	return sorted_doc_with_tfidf_score



def getNorm(docList1, docList2):
	norm1, norm2 = 0, 0
	for word in docList1:
		norm1 += docList1[word] ** 2
	for word in docList2:
		norm2 += docList2[word] ** 2
	
	res = math.sqrt(norm1 * norm2)
	return res



def getCosineSimilarity(sorted_doc_with_tfidf_score):
	cosine_similarity_matrix = []

	for docID1, docList1 in sorted_doc_with_tfidf_score:
		if len(docList1) == 0:
			continue
		score_vector = []
		for docID2, docList2 in sorted_doc_with_tfidf_score:
			if len(docList2) == 0:
				continue
			score = 0 
			# check if current word in the doocument
			for word in docList1:
				if word in docList2:
					# get numerator 
					score += docList1[word] * docList2[word]
					# get denominator
			norm = getNorm(docList1, docList2)
			score_vector.append(float(score) / norm)
		cosine_similarity_matrix.append(score_vector)
	return cosine_similarity_matrix			



def getBagofWords(sorted_doc_with_tfidf_score):
	words_list_without_stopwords = []
	for docID, wordList in sorted_doc_with_tfidf_score:
		for word in wordList:
			if word not in words_list_without_stopwords:
				words_list_without_stopwords.append(word)
	return words_list_without_stopwords



def getTFIDFMatrix(sorted_doc_with_tfidf_score, bagofwords_without_stopwords):
	TFIDF_matrix = []
	bagofwords_without_stopwords.sort()
	for each_doc in sorted_doc_with_tfidf_score:
		tfidf_vector = []
		for word in bagofwords_without_stopwords:
			if word in each_doc[1].keys():
				tfidf_vector.append(each_doc[1][word])
			else:
				tfidf_vector.append(0)
		TFIDF_matrix.append(tfidf_vector)
	return TFIDF_matrix




def getCluster(TFIDF_matrix):
	Z_single = linkage(TFIDF_matrix, 'single')
	Z_complete = linkage(TFIDF_matrix, 'complete')
	return Z_single, Z_complete




def getDocInfo():
	docID_Topic = {}
	corpusFile = open ("/homes/xu931/project2/reut2-subset.sgm", 'r')
	soup = BeautifulSoup(corpusFile, 'html.parser')
	reuters = soup.findAll('reuters')
	for every_doc in reuters:
		# extract the document ID 
		docID = int(every_doc['newid'])
		# extract the Topic
		documentTag = str(every_doc['topics'])

		if docID not in docID_Topic:
			docID_Topic[docID] = documentTag
	
	sorted_docID_Topic = sorted(docID_Topic.items(), key=operator.itemgetter(0), reverse=False)
	return sorted_docID_Topic



def getClusterDict(single_clustering, N, sorted_docID_Topic):
	# get the original hierachy
	inital_single_dict = {}

	current_cluster = N 
	for row in range(len(single_clustering)):
		clusters = []

		# check if the cluster already exists in the dict
		# if exsit -> merge all
		# if not -> add current
		first_cluster = single_clustering[row][0]
		second_cluster = single_clustering[row][1]

		if first_cluster in inital_single_dict or second_cluster in inital_single_dict:
			if first_cluster in inital_single_dict and second_cluster not in inital_single_dict:
				temp = inital_single_dict[first_cluster]
				temp.append(second_cluster)
				inital_single_dict[current_cluster] = temp
			if first_cluster not in inital_single_dict and second_cluster in inital_single_dict:
				temp = inital_single_dict[second_cluster][:]
				temp.append(first_cluster)
				inital_single_dict[current_cluster] = temp
			if first_cluster in inital_single_dict and second_cluster in inital_single_dict:
				temp1 = inital_single_dict[first_cluster]
				temp2 = inital_single_dict[second_cluster]
				inital_single_dict[current_cluster] = temp1 + temp2
		else:
			clusters.append(first_cluster)
			clusters.append(second_cluster)
			inital_single_dict[current_cluster] = clusters
		current_cluster += 1


	# map to new hiearchy dendrogram
	mapped_single_dict = {}
	new_start_point = N - 1
	for key, cluster in inital_single_dict.items():
		if new_start_point not in mapped_single_dict:
			mapped_single_dict[new_start_point] = cluster
		new_start_point -= 1


	doc_list = []
	# get a list of documents 
	for key, children in mapped_single_dict.items():
		for child in children:
			if child not in doc_list:
				doc_list.append(child)

	doc_list = sorted(doc_list)
	# map docList to a unique cluster
	# from lower index to higher
	index = N 
	map_doc_to_cluster = {}
	for doc in doc_list:
		map_doc_to_cluster[doc] = index
		index += 1

	# get the cluster for each document
	cluster_for_each_doc = {}
	for doc, mapped_cluster in map_doc_to_cluster.items():
		cluster_list = []
		for key, doclist in mapped_single_dict.items():
			if doc in doclist:
				cluster_list.append(key)
		cluster_list.append(mapped_cluster)
		cluster_for_each_doc[sorted_docID_Topic[int(doc)][0]] = cluster_list

	return cluster_for_each_doc

	# single_linkage_dict = {}
	# initial_cluster = (2 * N - 1) - N
	# for row in range(len(single_clustering)):
	# 	if initial_cluster <= 0:
	# 		break
	# 	else:
	# 		temp_list = []
	# 		temp_list.append(single_clustering[row][0])
	# 		temp_list.append(single_clustering[row][1])
	# 		single_linkage_dict[initial_cluster]  = temp_list
	# 	initial_cluster -= 1
	# return single_linkage_dict
	# while initial_cluster > 1:
	# 	if initial_cluster not in single_clustering:
	# 		single_clustering[initial_cluster] = []
	# 	for row in range(len(single_clustering)):
	# 		print (row)
	# 		# single_clustering[initial_cluster].append(single_clustering[row][0])
	# 		# single_clustering[initial_cluster].append(single_clustering[row][1])				
	# 	initial_cluster -= 1 
	
	# print single_linkage_dict
			
		

def writetoFile(mapping_single_clustering, fileName):
	f = open(fileName, "w")
	f.write("NEWID   clustersID\n")
	sorted_mapping_single_clustering = sorted(mapping_single_clustering.items(), key=operator.itemgetter(0), reverse=False)
	for doc, clusters in sorted_mapping_single_clustering:
		f.write(str(int(doc))+  "   ")
		for cluster in clusters:
			f.write(str(cluster) + " ")
		f.write("\n")




def parseTopic():

	# get docID with corresponding topics
	list_of_topics = {}
	corpusFile = open ("/homes/xu931/project2/reut2-subset.sgm", 'r')
	soup = BeautifulSoup(corpusFile, 'html.parser')
	reuters = soup.findAll('reuters')

	# get list of documents which have a topic
	list_of_docID = []
	for every_doc in reuters:
		# extract the document ID 
		docID = int(every_doc['newid'])
		# extract the topics
		topicTag = every_doc.findAll('topics')
		# ignore documents that do not have a topic
		for topic in topicTag:
			d = topic.findAll('d')
			if d:
				list_of_docID.append(docID)
				topic_list = []
				for i in d:
					topic_list.append(str(i.getText()))
				
				list_of_topics[docID] = topic_list

	# get a list of unique topics
	bagofuniquetopics = []
	for docID, topic_list in list_of_topics.items():
		for t in topic_list:
			if t not in bagofuniquetopics:
				bagofuniquetopics.append(t)

	return list_of_topics, bagofuniquetopics, sorted(list_of_docID)




def getInversedDoc(list_of_topics, bagofuniquetopics):
	reversedDocList = {}
	for t in bagofuniquetopics:
		docList = []
		for docID, topicList in list_of_topics.items():
			if t in topicList:
				docList.append(docID)
		reversedDocList[t] = docList

	return reversedDocList



def getScoreMatrix(list_of_topics, inversedDocList, list_of_docID):

	# form a initial matrix with 0 in every entry 
	scorematrix = []
	for i in range (len(list_of_docID)):
		scorevect = []
		for j in range(len(list_of_docID)):
			scorevect.append(0)
		scorematrix.append(scorevect)

	for topic, docList in inversedDocList.items():
		for doc1 in docList:
			for doc2 in docList:
				if doc1 != doc2:
					scorematrix[list_of_docID.index(doc1)][list_of_docID.index(doc2)] += 1 

	return scorematrix



def getIntersectUionHelper(docID1, docID2, mapping_single_clustering, mapping_complexe_clustering):
	single_docList1 = mapping_single_clustering[docID1]
	single_docList2 = mapping_single_clustering[docID2]
	single_intersect = len(set(single_docList1) & set(single_docList2))
	single_union = len(set(single_docList1) | set(single_docList2))
	

	complete_docList1 = mapping_complexe_clustering[docID1]
	complete_docList2 = mapping_complexe_clustering[docID2]
	complete_interesect = len(set(complete_docList1) & set(complete_docList2))
	complete_union = len(set(complete_docList1) | set(complete_docList2))
	return single_intersect, single_union, complete_interesect, complete_union



def minimizeMatrix(scorematrix, list_of_docID, mapping_single_clustering, mapping_complexe_clustering):
	single_matrix =[]
	complete_matrix = []
	for row in range(len(scorematrix)):
		for col in range(row, len(scorematrix[0])):
			if row != col:
				docID1 = list_of_docID[row]
				docID2 = list_of_docID[col]
				score = scorematrix[row][col]

				single_intersect, single_union, complete_interesect, complete_union = getIntersectUionHelper(docID1, docID2, mapping_single_clustering, mapping_complexe_clustering)
				single_sim_score = float(single_intersect) / single_union
				complete_sim_score = float(complete_interesect) / complete_union
			
				single_matrix.append([docID1, docID2, score, single_sim_score])
				complete_matrix.append([docID1, docID2, score, complete_sim_score])


	sorted_singel_matrix = sorted(single_matrix, key=lambda x:x[2])
	sorted_complete_matrix = sorted(complete_matrix, key=lambda x:x[2])
	return sorted_singel_matrix, sorted_complete_matrix



def evalMatrix(tupleScores):
	median_list = {}

	current_vector = tupleScores[0]
	temp = []
	temp.append(current_vector[3])
	median_list[current_vector[2]] = temp 
	for row in range(1, len(tupleScores)):
		if tupleScores[row][2] == current_vector[2]:
			median_list[tupleScores[row][2]].append(tupleScores[row][3])
		else:
			# update pointer to current row
			current_vector = tupleScores[row]
			temp = []
			temp.append(tupleScores[row][3])
			median_list[tupleScores[row][2]] = temp
	# find the median of each 
	topic_to_cluster = {}
	for common_topic, common_cluster in median_list.items():
		if len(common_cluster) == 1:
			topic_to_cluster[common_topic] = common_cluster[0]
		else:
			topic_to_cluster[common_topic] = statistics.median(common_cluster)

	return topic_to_cluster



def plotGraph(list_of_score):
	sorted_list = sorted(list_of_score.items())
	x, y = zip(*sorted_list)
	plt.plot(x, y)
	plt.show()


def main():

	### ------------------------------------------------- Preprocessing the corpus ----------------------------------------------- ###
	# parse the corpus, get N
	sorted_list_of_all_docs, N, bagofuniquewords = parseBody()
	# remove stop words 
	sorted_removedstopword_docs = removeStopwords(sorted_list_of_all_docs, N, bagofuniquewords)
	# stem words
	sorted_preprocessed_docs = stemDoc(sorted_removedstopword_docs)




	### ------------------------------------------------- TFIDF Score ----------------------------------------------- ###
	# calculate IDF for each word in each document
	IDF = getIDF(sorted_preprocessed_docs, N)
	# calculate TFIDF for each word in each document
	sorted_doc_with_tfidf_score = getTFIDF(sorted_preprocessed_docs, IDF)





	### ------------------------------------------------- Cosine Similarity ------------------------------------------------- ### 
	# calculate cosine similiarty for each doc
	cs_matrix = getCosineSimilarity(sorted_doc_with_tfidf_score)
	#get new bag of words (after removing stopwrod)
	bagofwords_without_stopwords = getBagofWords(sorted_doc_with_tfidf_score)
	# get TFIDF matrix according to bag of words
	TFIDF_matrix = getTFIDFMatrix(sorted_doc_with_tfidf_score, bagofwords_without_stopwords)




	### ------------------------------------------------- Single & Complete Cluster ----------------------------------------------- ###
	# get single and complete linkage cluster
	single_clustering, complete_clustering = getCluster(TFIDF_matrix)
	# get doc info
	sorted_docID_Topic = getDocInfo()
	# mapping each doc to its corresponding single cluster according to the matrix
	mapping_single_clustering = getClusterDict(single_clustering, len(TFIDF_matrix), sorted_docID_Topic)
	# mapping each doc to its corresponding complete cluster accoding to the matrix
	mapping_complexe_clustering = getClusterDict(complete_clustering, len(TFIDF_matrix), sorted_docID_Topic)
	# write to single.txt, complete.txt
	single_txt = "single.txt"
	complete_txt = "complete.txt"
	writetoFile(mapping_single_clustering, single_txt)
	writetoFile(mapping_complexe_clustering, complete_txt)
	# # plot dendrogram
	# fig = plt.figure(figsize=(25, 10))
	# dn = dendrogram(single_clustering)
	# plt.show()
	# plt.close()



	### ------------------------------------------------- Evaluation ----------------------------------------------- ###
	# parse the corpus: get {DOCID:[topic1, topic2,...]}
	list_of_topics, bagofuniquetopics, list_of_docID= parseTopic()	
	# inversed doc list: get the documents which contains the topic
	inversedDocList = getInversedDoc(list_of_topics, bagofuniquetopics)
	# form the score matrix for each document
	scoreMatrix = getScoreMatrix(list_of_topics, inversedDocList, list_of_docID)
	# get tuple scores from the matrix 
	singletupleScores, completetupleScores = minimizeMatrix(scoreMatrix, list_of_docID, mapping_single_clustering, mapping_complexe_clustering)
	# interpret the matrix by the (num_of_common_topics : num_of_common_clusters)
	# percentage = num_of_common_topics / num_of_common_clusters (0 if num_of_common_clusters is 0)
	list_of_percentage_single = evalMatrix(singletupleScores)
	list_of_percentage_complete = evalMatrix(completetupleScores)
	# plot the graph for relation between common_topics & common_clusters
	plotGraph(list_of_percentage_single)
	plotGraph(list_of_percentage_complete)




main()
