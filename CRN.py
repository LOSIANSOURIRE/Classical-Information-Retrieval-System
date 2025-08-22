from util import *



class CRNInformationRetrieval():

    def __init__(self):
        self.index = None
        self.docIDs = None
        self.docs = None
        self.index_list=None
        self.term_doc_matrix=None
        
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

        self.index_list=list(index.keys())
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
		# print("D",D)
        df=self.doc_freq()
		# print ("self.doc_freq",df)
        tf_w={}
        for pdf in self.docIDs:
            tf_w[pdf]={}
            for i in self.index.keys():
                if(pdf in self.index[i].keys()):
                    tf_w[pdf][i]=self.index[i][pdf]*math.log10(D/df[i])

        return tf_w
    
    
    def doc_dict_to_numpy(self,documents):
        
        tf_idf_dict=self.tf_idf_doc_vec()

        #Convert documnet dictionary to numpy array
        term_doc_matrix=np.zeros((len(self.index.keys()),len(documents))) #Make a term document matrix

        for termid in range (len(self.index.keys())):
            term=self.index_list[termid]
            
            for docs in self.index[term].keys():
                term_doc_matrix[termid][docs-1]=tf_idf_dict[docs][term]
        
        return term_doc_matrix
    

   
    
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
    


    def query_dict_to_numpy(self,Querylists):
        
        tf_idf_dict=self.tf_idf_query_vec(Querylists,self.doc_freq())
        #Convert documnet dictionary to numpy array
        term_query_matrix=np.zeros((len(self.index.keys()),len(Querylists))) #Make a term document matrix

        for termid in range (len(self.index.keys())):
            term=self.index_list[termid]
            
            for query in range(1,len(Querylists)+1):
                if term in tf_idf_dict[query].keys():
                    term_query_matrix[termid][query-1]=tf_idf_dict[query][term]
        
        return term_query_matrix


    def term_term_simlarity_matrix(self,args):
        term_term_matrix=np.ones((len(self.index.keys()),len(self.index.keys())))

        c=0
        for i in range(len(self.index.keys())):
            for j in range(i):

                print(c)
                c+=1
                p_occ_i=len(self.index[self.index_list[i]].keys())/len(self.docs)
                p_occ_j=len(self.index[self.index_list[j]].keys())/len(self.docs)
                p_occ_i_j=len(set(self.index[self.index_list[i]].keys()).intersection(set(self.index[self.index_list[j]].keys())))/len(self.docs)
                if p_occ_i==0 or p_occ_j==0 or p_occ_i_j==0:
                    term_term_matrix[i][j]=0
                    term_term_matrix[j][i]=0
                else:
                    term_term_matrix[i][j]=math.log2(p_occ_i_j/(p_occ_i*p_occ_j))
                    term_term_matrix[j][i]=math.log2(p_occ_i_j/(p_occ_i*p_occ_j))

        json.dump(term_term_matrix.tolist(), open(args + "term_term_similarity_matrix.txt", 'w'))
        return term_term_matrix

    def doc_cos_similarity(self,query, documents):
        return cosine_similarity(query.T, documents.T)

    def rank(self, queries,args):
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

        self.term_doc_matrix=self.doc_dict_to_numpy(self.docs)
    
        try:
            term_term_matrix=np.array(json.load(open(args + "term_term_similarity_matrix.txt")))
        except:
            term_term_matrix=self.term_term_simlarity_matrix(args)

        doc_IDs_ordered = []
        
        start_time = time.time()
        
        self.term_query_matrix=self.query_dict_to_numpy(queries)
        modified_query=np.dot(term_term_matrix,self.term_query_matrix)

        query_doc_cos_similarity=self.doc_cos_similarity(modified_query,np.dot(term_term_matrix,self.term_doc_matrix)) 


        for query in query_doc_cos_similarity:
            
            sorted_indices = np.argsort(query)[::-1]
            doc_IDs_ordered.append(sorted_indices+1)

        if(query_doc_cos_similarity.shape[0]==1):  #For custom query
            if(max(query)==0):
                return -1
        
        end_time = time.time()
        print("Time taken for retrieval",end_time-start_time)
        return doc_IDs_ordered