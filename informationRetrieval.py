from util import *



class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docIDs = None
		self.docs = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		self.docIDs = docIDs
		self.docs = docs
		index = {}
		i=0
		
		for doc in docs:
			for lines in doc:
				for words in lines:

					
					if(words not in index.keys()):
						index[words]={}
		
					if (docIDs[i] not in index[words].keys()):
						index[words][docIDs[i]]=0
						
					c=lines.count(words)
					if(c!=0):
						index[words][docIDs[i]]+=1
			i+=1

		self.index = index
	
	def doc_freq(self):
		df={}
		for i in self.index.keys():
			df[i]=0
			for pdf in self.docIDs:
				if pdf in self.index[i].keys():
					df[i]=df[i]+1

		return df

	def tf_idf_doc_vec(self):
		D=len(self.docs)
		df=self.doc_freq()

		tf_w={}
		for pdf in self.docIDs:
			tf_w[pdf]={}
			for i in self.index.keys():
				if(pdf in self.index[i].keys()):

					tf_w[pdf][i]=self.index[i][pdf]*math.log10(D/df[i])
  
		return tf_w





	def query_freq(self,Querylists):

		qf={}
		c=1

		for querylist in Querylists:
			qf[c]={}
			for sentence in querylist:
				for i in sentence:
					if i not in qf[c].keys():
						qf[c][i]=0
					
					qf[c][i]=qf[c][i]+1
			c+=1
		return qf
	
	
	def tf_idf_query_vec(self,Querylists,df):

		D=len(self.docs)
		qf=self.query_freq(Querylists)
		tf_q={}
		c=1

		for querylist in Querylists:
			tf_q[c]={}
			for sentence in querylist:
				for i in sentence:
					if(i in self.index.keys()):
						if qf[c][i]!=0:
							tf_q[c][i]=qf[c][i]*math.log10(D/df[i])
						
					else:
						tf_q[c][i]=0
			c+=1
  
		return tf_q

	def mod_d(self,vec):
		c=0
		for i in vec:
			c=c+vec[i]*vec[i]

		return math.sqrt(c)	

	def doc_cos_similarity(self,Querylists,q,w):
		cos={}
		c=1

		for query in Querylists:
			cos[c]={}
			for sentence in query:
				for word in sentence:
					if word in self.index.keys():		
						for docs in self.index[word].keys():
							if docs not in cos[c].keys():
								cos[c][docs]=0
							
							cos[c][docs]+=q[c][word]*w[docs][word]
					else:
						if docs not in cos[c].keys():
							cos[c][docs]=0
						cos[c][docs]+=0
			c+=1

		for query in range (1,len(Querylists)+1):
			for docs in self.docIDs:
				if docs in cos[query].keys():
					cos[query][docs]=cos[query][docs]/(self.mod_d(w[docs])*self.mod_d(q[query]))
		
		return cos






	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""


		doc_IDs_ordered = []
		w=self.tf_idf_doc_vec()
		df=self.doc_freq()

		start_time = time.time()

		q=self.tf_idf_query_vec(queries,df)
		query_doc_cos_similarity=self.doc_cos_similarity(queries,q,w)
		for query in range(1,len(queries)+1):

			doc_IDs_ordered.append(sorted(query_doc_cos_similarity[query], key=query_doc_cos_similarity[query].get, reverse=True)[:10])
		

		end_time = time.time()
		print("Time taken for retrieval",end_time-start_time)
		return doc_IDs_ordered