from util import *
class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		k_docs = query_doc_IDs_ordered[:k]
		relevant_k_docs = [doc for doc in k_docs if doc in true_doc_IDs]
		precision = len(relevant_k_docs) / k

		
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k,args):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""
		meanPrecision = -1

		#Fill in code here

		qrelevant = []
		for i in query_ids:
			qrelevant.append([int(query['id']) for query in qrels if int(query['query_num']) == i])
		# print(qrelevant)
		precision_sum=0
		precision=[]
		for i in range(len(doc_IDs_ordered)):
			precision_sum+=self.queryPrecision(doc_IDs_ordered[i], query_ids[i], qrelevant[i], k)
			precision.append(self.queryPrecision(doc_IDs_ordered[i], query_ids[i], qrelevant[i], k))
		if k==6:
			if args.LSA:
				json.dump(precision, open(args.out_folder + "precision_LSA.txt", 'w'))
			elif args.CRN:
				json.dump(precision, open(args.out_folder + "precision_CRN.txt", 'w'))
			else:
				json.dump(precision, open(args.out_folder + "precision_VSM.txt", 'w'))
		

		meanPrecision = precision_sum/len(doc_IDs_ordered)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		k_docs = query_doc_IDs_ordered[:k]
		relevant_k_docs = [doc for doc in k_docs if doc in true_doc_IDs]
		recall = len(relevant_k_docs) / len(true_doc_IDs)


		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k,args):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		qrelevant = []
		for i in query_ids:
			qrelevant.append([int(query['id']) for query in qrels if int(query['query_num']) == i])

		#Fill in code here
		recall_sum=0
		recall=[]
		for i in range(len(doc_IDs_ordered)):
			recall_sum+=self.queryRecall(doc_IDs_ordered[i], query_ids[i], qrelevant[i], k)
			recall.append(self.queryRecall(doc_IDs_ordered[i], query_ids[i], qrelevant[i], k))
		if k==6:
			if args.LSA:
				json.dump(recall, open(args.out_folder + "recall_LSA.txt", 'w'))
			elif args.CRN:
				json.dump(recall, open(args.out_folder + "recall_CRN.txt", 'w'))
			else:
				json.dump(recall, open(args.out_folder + "recall_VSM.txt", 'w'))


		meanRecall = recall_sum/len(doc_IDs_ordered)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if precision + recall == 0:
			fscore = 0
		else:
			fscore = 1.25 * (precision * recall) / (0.25 * precision + recall)

		

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k,args):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1
		qrelevant = []
		for i in query_ids:
			qrelevant.append([int(query['id']) for query in qrels if int(query['query_num']) == i])
		#Fill in code here
		Fscore=[]
		fscore_sum=0
		for i in range(len(doc_IDs_ordered)):
			fscore_sum+=self.queryFscore(doc_IDs_ordered[i], query_ids[i], qrelevant[i], k)
			Fscore.append(self.queryFscore(doc_IDs_ordered[i], query_ids[i], qrelevant[i], k))
		if k==6:
			if args.LSA:
				json.dump(Fscore, open(args.out_folder + "Fscore_LSA.txt", 'w'))
			elif args.CRN:
				json.dump(Fscore, open(args.out_folder + "Fscore_CRN.txt", 'w'))
			else:
				json.dump(Fscore, open(args.out_folder + "Fscore_VSM.txt", 'w'))
			



		meanFscore = fscore_sum/len(doc_IDs_ordered)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs,rel, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		dcg=0
		idcg=0


		for i in range(min(k,len(rel))):
			if query_doc_IDs_ordered[i] in true_doc_IDs:

				rel_index=true_doc_IDs.index(query_doc_IDs_ordered[i])
				dcg+=(2**(5-rel[rel_index])-1)/(math.log2(i+2))

		rel.sort(reverse=False)
		for i in range(min(len(rel),k)):
			idcg+=(2**(5-rel[i])-1)/(math.log2(i+2))

		if idcg!=0:
			nDCG = dcg/idcg
		else :
			nDCG = 0
		
		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k,args):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		qrelevant = []
		rel_pos=[]
		for i in query_ids:
			qrelevant.append([int(query['id']) for query in qrels if int(query['query_num']) == i])
			rel_pos.append([int(query['position']) for query in qrels if int(query['query_num']) == i])
		# #Fill in code here	
		ndcg_sum=0
		nDCG=[]
		for i in range(len(doc_IDs_ordered)):
			ndcg_sum+=self.queryNDCG(doc_IDs_ordered[i], query_ids[i], qrelevant[i],rel_pos[i], k)
			nDCG.append(self.queryNDCG(doc_IDs_ordered[i], query_ids[i], qrelevant[i],rel_pos[i], k))
		if k==6:
			if args.LSA:
				json.dump(nDCG, open(args.out_folder + "nDCG_LSA.txt", 'w'))
			elif args.CRN:
				json.dump(nDCG, open(args.out_folder + "nDCG_CRN.txt", 'w'))
			else:
				json.dump(nDCG, open(args.out_folder + "nDCG_VSM.txt", 'w'))
			
		meanNDCG = ndcg_sum/len(doc_IDs_ordered)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of intege.rs denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1
		#Fill in code here
		precision_sum=0
		relevant_docs=0
		
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				relevant_docs+=1
				precision_sum+=relevant_docs/(i+1)
		
		if  relevant_docs!=0:
			avgPrecision=precision_sum/relevant_docs
		else:
			avgPrecision=0

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k,args):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1


		qrelevant = []
		for i in query_ids:
			qrelevant.append([int(query['id']) for query in q_rels if int(query['query_num']) == i])
		#Fill in code here
		avgPrecision_sum=0
		avgPrecision=[]
		for i in range(len(doc_IDs_ordered)):
			avgPrecision_sum+=self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], qrelevant[i], k)
			avgPrecision.append(self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], qrelevant[i], k))

		if k==6:
			if args.LSA:
				json.dump(avgPrecision, open(args.out_folder + "avgPrecision_LSA.txt", 'w'))
			elif args.CRN:
				json.dump(avgPrecision, open(args.out_folder + "avgPrecision_CRN.txt", 'w'))
			else:
				json.dump(avgPrecision, open(args.out_folder + "avgPrecision_VSM.txt", 'w'))

		meanAveragePrecision = avgPrecision_sum/len(doc_IDs_ordered)
		
		return meanAveragePrecision

class Hypothesis_Testing():

	def t_test(self, list1, list2):
		"""
		Performs a t-test to determine if the two lists of numbers are significantly different

		Parameters
		----------
		arg1 : list
			A list of numbers
		arg2 : list
			A list of numbers

		Returns
		-------
		float
			The p-value
		"""

		p_value = -1

		#Fill in code here
		t_stat, p_value = stats.ttest_ind(list1, list2)

		return p_value
	
	def load_metrics(self,args):
		"""
		Loads the metrics from the output files

		Returns
		-------
		list
			A list of lists where the ith sub-list is a list of metrics for the ith model
		"""

		metrics_precision = []
		metrics_avgPrecision = []
		metrics_recall = []
		metrics_Fscore = []
		metrics_nDCG = []
	

		metrics_avgPrecision.append(json.load(open(args.out_folder+"avgPrecision_VSM.txt")))
		metrics_avgPrecision.append(json.load(open(args.out_folder+"avgPrecision_LSA.txt")))
		metrics_avgPrecision.append(json.load(open(args.out_folder+"avgPrecision_CRN.txt")))

		sns.set(style="whitegrid")
		sns.kdeplot(metrics_avgPrecision[0], label="VSM")
		sns.kdeplot(metrics_avgPrecision[2], label="CRN")
		plt.xlabel("AveragePrecision")
		plt.ylabel("Density")
		plt.title("Density plot of AveragePrecision@6")
		plt.legend()
		plt.show()
		plt.savefig(args.out_folder+"AveragePrecision_density_plot_CRN.png")
		plt.close()

		sns.set(style="whitegrid")
		sns.kdeplot(metrics_avgPrecision[0], label="VSM")
		sns.kdeplot(metrics_avgPrecision[1], label="LSA")
		plt.xlabel("AveragePrecision")
		plt.ylabel("Density")
		plt.title("Density plot of AveragePrecision@6")
		plt.legend()
		plt.show()
		plt.savefig(args.out_folder+"AveragePrecision_density_plot_LSA.png")
		plt.close()



		metrics_precision.append(json.load(open(args.out_folder+"precision_VSM.txt")))
		metrics_precision.append(json.load(open(args.out_folder+"precision_LSA.txt")))
		metrics_precision.append(json.load(open(args.out_folder+"precision_CRN.txt")))

		sns.set(style="whitegrid")
		sns.kdeplot(metrics_precision[0], label="VSM")
		sns.kdeplot(metrics_precision[1], label="LSA")
		plt.xlabel("Precision")
		plt.ylabel("Density")
		plt.title("Density plot of Precision@6")
		plt.legend()
		plt.show()
		plt.savefig(args.out_folder+"Precision_density_plot_LSA.png")
		plt.close()

		sns.set(style="whitegrid")
		sns.kdeplot(metrics_precision[0], label="VSM")
		sns.kdeplot(metrics_precision[2], label="CRN")
		plt.xlabel("Precision")
		plt.ylabel("Density")
		plt.title("Density plot of Precision@6")
		plt.legend()
		plt.show()
		plt.savefig(args.out_folder+"Precision_density_plot_CRN.png")
		plt.close()

		metrics_recall.append(json.load(open(args.out_folder+"recall_VSM.txt")))
		metrics_recall.append(json.load(open(args.out_folder+"recall_LSA.txt")))
		metrics_recall.append(json.load(open(args.out_folder+"recall_CRN.txt")))

		sns.set(style="whitegrid")
		sns.kdeplot(metrics_recall[0], label="VSM")
		sns.kdeplot(metrics_recall[2], label="CRN")
		plt.xlabel("Recall")
		plt.ylabel("Density")
		plt.title("Density plot of Recall@6")
		plt.legend()
		plt.show()
		plt.savefig(args.out_folder+"Recall_density_plot_CRN.png")
		plt.close()

		sns.set(style="whitegrid")
		sns.kdeplot(metrics_recall[0], label="VSM")
		sns.kdeplot(metrics_recall[1], label="LSA")
		plt.xlabel("Recall")
		plt.ylabel("Density")
		plt.title("Density plot of Recall@6")
		plt.legend()
		plt.show()
		plt.savefig(args.out_folder+"Recall_density_plot_LSA.png")
		plt.close()

		metrics_Fscore.append(json.load(open(args.out_folder+"Fscore_VSM.txt")))
		metrics_Fscore.append(json.load(open(args.out_folder+"Fscore_LSA.txt")))
		metrics_Fscore.append(json.load(open(args.out_folder+"Fscore_CRN.txt")))


		sns.set(style="whitegrid")
		sns.kdeplot(metrics_Fscore[0], label="VSM")
		sns.kdeplot(metrics_Fscore[1], label="LSA")
		plt.xlabel("Fscore")
		plt.ylabel("Density")
		plt.title("Density plot of Fscore@6")
		plt.legend()
		plt.show()
		plt.savefig(args.out_folder+"Fscore_density_plot_LSA.png")
		plt.close()

		sns.set(style="whitegrid")
		sns.kdeplot(metrics_Fscore[0], label="VSM")
		sns.kdeplot(metrics_Fscore[2], label="CRN")
		plt.xlabel("Fscore")
		plt.ylabel("Density")
		plt.title("Density plot of Fscore@6")
		plt.legend()
		plt.show()
		plt.savefig(args.out_folder+"Fscore_density_plot_CRN.png")
		plt.close()

		metrics_nDCG.append(json.load(open(args.out_folder+"nDCG_VSM.txt")))
		metrics_nDCG.append(json.load(open(args.out_folder+"nDCG_LSA.txt")))
		metrics_nDCG.append(json.load(open(args.out_folder+"nDCG_CRN.txt")))

		sns.set(style="whitegrid")
		sns.kdeplot(metrics_nDCG[0], label="VSM")
		sns.kdeplot(metrics_nDCG[2], label="CRN")
		plt.xlabel("nDCG")
		plt.ylabel("Density")
		plt.title("Density plot of nDCG@6")
		plt.legend()
		plt.show()
		plt.savefig(args.out_folder+"nDCG_density_plot_CRN.png")
		plt.close()

		sns.set(style="whitegrid")
		sns.kdeplot(metrics_nDCG[0], label="VSM")
		sns.kdeplot(metrics_nDCG[1], label="LSA")
		plt.xlabel("nDCG")
		plt.ylabel("Density")
		plt.title("Density plot of nDCG@6")
		plt.legend()
		plt.show()
		plt.savefig(args.out_folder+"nDCG_density_plot_LSA.png")
		plt.close()


		print("CRN vs VSM")
		print("Precision:", self.t_test(metrics_precision[0], metrics_precision[2]))
		print("Recall:", self.t_test(metrics_recall[0], metrics_recall[2]))
		print("Fscore:", self.t_test(metrics_Fscore[0], metrics_Fscore[2]))
		print("nDCG:", self.t_test(metrics_nDCG[0], metrics_nDCG[2]))
		print("MAP:", self.t_test(metrics_avgPrecision[0], metrics_avgPrecision[2]))

		print("VSM vs LSA")
		print("Precision: ", self.t_test(metrics_precision[0], metrics_precision[1]))
		print("Recall: ", self.t_test(metrics_recall[0], metrics_recall[1]))
		print("Fscore: ", self.t_test(metrics_Fscore[0], metrics_Fscore[1]))
		print("nDCG: ", self.t_test(metrics_nDCG[0], metrics_nDCG[1]))
		print("MAP: ", self.t_test(metrics_avgPrecision[0], metrics_avgPrecision[1]))

		print("LSA and CRN")
		print("Precision: ", self.t_test(metrics_precision[1], metrics_precision[2]))
		print("Recall: ", self.t_test(metrics_recall[1], metrics_recall[2]))
		print("Fscore: ", self.t_test(metrics_Fscore[1], metrics_Fscore[2]))
		print("nDCG: ", self.t_test(metrics_nDCG[1], metrics_nDCG[2]))
		print("MAP: ", self.t_test(metrics_avgPrecision[1], metrics_avgPrecision[2]))

