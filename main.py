from util import *
start_time = time.time()

from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from Vocabulary import CandidateSelection
from informationRetrieval import InformationRetrieval
from LSA import LSAInformationRetrieval
from CRN import CRNInformationRetrieval
from evaluation import Evaluation
from evaluation import Hypothesis_Testing

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()
		self.candidate_Selection=CandidateSelection()

		if args.LSA:
			self.informationRetriever = LSAInformationRetrieval()
		elif args.CRN:
			self.informationRetriever = CRNInformationRetrieval()
		else:
			self.informationRetriever = InformationRetrieval()

		self.autocomplete = InformationRetrieval()
		self.evaluator = Evaluation()
		self.hypothesis_tester = Hypothesis_Testing()

	def segmentSentences(self, text):
		"""
		Return the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Return the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Return the required stemmer/lemmatizer
		"""
		if self.args.stemmer == "porter":
			return self.inflectionReducer.reduce(text)
		elif self.args.stemmer == "lemmatizer":
			return self.inflectionReducer.lemmatize_text(text)

	def removeStopwords(self, text):
		"""
		Return the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)


	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		
		# Segment docs

		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))


		
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))


		# Remove stopwords from docs(using NLTK)
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)			
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs

		return preprocessedDocs

	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for each query
		if self.args.CRN:
			doc_IDs_ordered = self.informationRetriever.rank(processedQueries,self.args.out_folder)
		else:
			doc_IDs_ordered = self.informationRetriever.rank(processedQueries)


		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k,self.args)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k,self.args)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k,self.args)
			fscores.append(fscore)
			print("Precision, Recall and F-score @ " +  
				str(k) + " : " + str(precision) + ", " + str(recall) + 
				", " + str(fscore))
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k,self.args)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k,self.args)
			nDCGs.append(nDCG)
			print("MAP, nDCG @ " +  
				str(k) + " : " + str(MAP) + ", " + str(nDCG))

		# Plot the metrics and save plot 
		if args.LSA:
			json.dump(precisions, open(self.args.out_folder + "precision_lsa.txt", 'w'))
			json.dump(recalls, open(self.args.out_folder + "recall_lsa.txt", 'w'))
			json.dump(fscores, open(self.args.out_folder + "fscore_lsa.txt", 'w'))
			json.dump(MAPs, open(self.args.out_folder + "MAP_lsa.txt", 'w'))
			json.dump(nDCGs, open(self.args.out_folder + "nDCG_lsa.txt", 'w'))
		elif args.CRN:
			json.dump(precisions, open(self.args.out_folder + "precision_crn.txt", 'w'))
			json.dump(recalls, open(self.args.out_folder + "recall_crn.txt", 'w'))
			json.dump(fscores, open(self.args.out_folder + "fscore_crn.txt", 'w'))
			json.dump(MAPs, open(self.args.out_folder + "MAP_crn.txt", 'w'))
			json.dump(nDCGs, open(self.args.out_folder + "nDCG_crn.txt", 'w'))
		else:
			json.dump(precisions, open(self.args.out_folder + "precision.txt", 'w'))
			json.dump(recalls, open(self.args.out_folder + "recall.txt", 'w'))
			json.dump(fscores, open(self.args.out_folder + "fscore.txt", 'w'))
			json.dump(MAPs, open(self.args.out_folder + "MAP.txt", 'w'))
			json.dump(nDCGs, open(self.args.out_folder + "nDCG.txt", 'w'))
		# plt.plot(range(1, 11), precisions, label="Precision")
		# plt.plot(range(1, 11), recalls, label="Recall")
		# plt.legend()
		# plt.title("Evaluation Metrics - Cranfield Dataset - Precision and Recall")
		# plt.xlabel("k")
		# plt.savefig(args.out_folder + "eval_plot_precision_and_recall.png")
		# plt.show()

		# plt.plot(range(1, 11), fscores, label="F-Score")
		# plt.legend()
		# plt.title("Evaluation Metrics - Cranfield Dataset- F-Score")
		# plt.xlabel("k")
		# plt.savefig(args.out_folder + "eval_plot_f-score.png")
		# plt.show()

		# plt.plot(range(1, 11), MAPs, label="MAP")
		# plt.legend()
		# plt.title("Evaluation Metrics - Cranfield Dataset- MAP")
		# plt.xlabel("k")
		# plt.savefig(args.out_folder + "eval_plot_MAP.png")
		# plt.show()

		# plt.plot(range(1, 11), nDCGs, label="nDCG")
		# plt.legend()
		# plt.title("Evaluation Metrics - Cranfield Dataset- nDCG")
		# plt.xlabel("k")
		# plt.savefig(args.out_folder + "eval_plot_nDCG.png")
		# plt.show()

		plt.plot(range(1, 11), precisions, label="Precision")
		plt.plot(range(1, 11), recalls, label="Recall")
		plt.plot(range(1, 11), fscores, label="F-Score")
		plt.plot(range(1, 11), MAPs, label="MAP")
		plt.plot(range(1, 11), nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		plt.savefig(args.out_folder + "eval_plot.png")
		plt.show()

		"""Plot two of the metrics in a single plot"""
		# precision_lsa=json.load(open(self.args.out_folder + "precision_lsa.txt", 'r'))
		# recall_lsa=json.load(open(self.args.out_folder + "recall_lsa.txt", 'r'))
		# fscore_lsa=json.load(open(self.args.out_folder + "fscore_lsa.txt", 'r'))
		# MAP_lsa=json.load(open(self.args.out_folder + "MAP_lsa.txt", 'r'))
		# nDCG_lsa=json.load(open(self.args.out_folder + "nDCG_lsa.txt", 'r'))

		# precision_crn=json.load(open(self.args.out_folder + "precision_crn.txt", 'r'))
		# recall_crn=json.load(open(self.args.out_folder + "recall_crn.txt", 'r'))
		# fscore_crn=json.load(open(self.args.out_folder + "fscore_crn.txt", 'r'))
		# MAP_crn=json.load(open(self.args.out_folder + "MAP_crn.txt", 'r'))
		# nDCG_crn=json.load(open(self.args.out_folder + "nDCG_crn.txt", 'r'))


		# plt.figure(figsize=(10, 10))
		# plt.plot(range(1, 11), precisions, label="Precision",color='red')
		# plt.plot(range(1, 11), recalls, label="Recall",color='blue')
		# plt.plot(range(1, 11), fscores, label="F-Score",color='green')
		# plt.plot(range(1, 11), MAPs, label="MAP",color='yellow')
		# plt.plot(range(1, 11), nDCGs, label="nDCG",color='cyan')
		# plt.plot(range(1, 11), precision_lsa, label="Precision_LSA",linestyle='dashed',color='red')
		# plt.plot(range(1, 11), recall_lsa, label="Recall_LSA",linestyle='dashed',color='blue')
		# plt.plot(range(1, 11), fscore_lsa, label="F-Score_LSA",linestyle='dashed',color='green')
		# plt.plot(range(1, 11), MAP_lsa, label="MAP_LSA",linestyle='dashed',color='yellow')
		# plt.plot(range(1, 11), nDCG_lsa, label="nDCG_LSA",linestyle='dashed',color='cyan')
		# plt.legend()
		# plt.title("Evaluation Metrics - Cranfield Dataset")
		# plt.xlabel("k")
		# plt.savefig(args.out_folder + "eval_plot.png")
		# plt.show()

		# plt.plot(range(1, 11), precisions, label="Precision",color='red')
		# plt.plot(range(1, 11), recalls, label="Recall",color='blue')
		# plt.plot(range(1, 11), fscores, label="F-Score",color='green')
		# plt.plot(range(1, 11), MAPs, label="MAP",color='yellow')
		# plt.plot(range(1, 11), nDCGs, label="nDCG",color='cyan')
		# plt.plot(range(1, 11), precision_crn, label="Precision_CRN",linestyle='dashed',color='red')
		# plt.plot(range(1, 11), recall_crn, label="Recall_CRN",linestyle='dashed',color='blue')
		# plt.plot(range(1, 11), fscore_crn, label="F-Score_CRN",linestyle='dashed',color='green')
		# plt.plot(range(1, 11), MAP_crn, label="MAP_CRN",linestyle='dashed',color='yellow')
		# plt.plot(range(1, 11), nDCG_crn, label="nDCG_CRN",linestyle='dashed',color='cyan')
		# plt.legend()
		# plt.title("Evaluation Metrics - Cranfield Dataset")
		# plt.xlabel("k")
		# plt.savefig(args.out_folder + "eval_plot.png")
		# plt.show()


		"""Hypothesis Testing"""
		# print("Hypothesis Testing")
		# self.hypothesis_tester.load_metrics(self.args)


	def preprocess_autocomplete_Queries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_custom_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_custom_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_custom_queries.txt", 'w'))

		preprocessedQueries = reducedQueries
		return preprocessedQueries


	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		#Get query
		print("Enter query below")
		query = input()

		""""Autocomplete the query using the inbuilt query corpus"""
		try:
			#Read inbuilt query corpus
			queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
			query_ids, queries = [item["query number"] for item in queries_json], \
									[item["query"] for item in queries_json]
			
			# Process queries 
			processedQueries = self.preprocess_autocomplete_Queries(queries)
			auto_complete_query=self.preprocess_autocomplete_Queries([query])
			self.autocomplete.buildIndex(processedQueries, query_ids)
			# Rank the best query for the inputted query
			probable_query_list=self.autocomplete.rank(auto_complete_query)

			try:

				for i in range (min(5,len(probable_query_list[0]))):
					print(f"{i+1}: {queries[probable_query_list[0][i]-1]}")

				pos=input("Choose the query you are searching for:")
				probab_query = probable_query_list[0][int(pos)-1]
				
			except:
	
				probab_query = probable_query_list[0][0]
				print("Most probable query_hi:",queries[probab_query-1])
			print("Choosen query:",queries[probab_query-1])
			query =queries[probab_query-1]
		except:
			query=query
			print("No relevant query found so choosen query is:",query)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)
		
		# Process documents
		processedQuery = self.preprocessQueries([query])[0]
		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for the query
		doc_IDs_ordered=[]
		try:
			doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]
			# Print the IDs of first five documents
			print("\nTop five document IDs : ")
			for id_ in doc_IDs_ordered[:5]:
				print(id_)
		except:
			print("No relevant document found")

		



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [punkt|naive]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [ptb|naive]")
	parser.add_argument('-stemmer', default = "porter"
					 	, help = "Stemmer Type [porter|lemmatizer]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	parser.add_argument('-LSA', action = "store_true", 
						help = "InformationRetrieval")
	parser.add_argument('-CRN', action = "store_true", 
						help = "InformationRetrieval")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
	
end_time = time.time()
# Calculate elapsed time
elapsed_time = end_time - start_time

# Print the run time
print("Elapsed time (seconds):", elapsed_time)