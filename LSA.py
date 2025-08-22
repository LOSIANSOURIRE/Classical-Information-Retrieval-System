from util import *



class LSAInformationRetrieval():

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
        t=True
        for termid in range (len(self.index.keys())):
            term=self.index_list[termid]
            
            for docs in self.index[term].keys():
                if t:
                    t=False
                term_doc_matrix[termid][docs-1]=tf_idf_dict[docs][term]
        
        return term_doc_matrix
    

    def cumulative_explained_variance_ratio(self,sigma,k):
        """
        Compute the cumulative explained variance ratio given the singular values.

        Parameters:
        - sigma: numpy array containing the singular values.
        - k: number of components to consider.

        Returns:
        - cumulative_explained_variance_ratio: numpy array containing the cumulative explained variance ratio up to each component.
        """
        # Compute the squared singular values
        sigma_squared = sigma**2
        cumulative_explained_variance_ratio = np.sum(sigma_squared[:k]) / np.sum(sigma_squared)
        return cumulative_explained_variance_ratio

    def LSA(self):
        """
        Perform Latent Semantic Analysis (LSA) after finding the best k value.

        Returns:
        - lsa_matrix: numpy array containing the LSA matrix.
        -U_reduce: numpy array containing the reduced U matrix.
        -Sigma_reduce: numpy array containing the reduced Sigma matrix.
        -Vt_reduce: numpy array containing the reduced Vt matrix.

        """

        # Perform Singular Value Decomposition (SVD)
        U, Sigma, Vt = np.linalg.svd(self.term_doc_matrix)

        num_concepts=0
        sigma_=np.diag(Sigma).diagonal()

        """ Plot explained_variance_ratio to get the elbow point """

        # sigma_squared = sigma_**2
        # explained_variance_ratio =sigma_squared[50:] / np.sum(sigma_squared)
        # rate_of_change = np.diff(explained_variance_ratio)
        # knee_point = np.argmax(rate_of_change) + 1
        #num_concepts=knee_point
    
        # # Plot singular values to get the elbow point
        # plt.plot(explained_variance_ratio, marker='o', color='blue')
        # plt.title("Scree Plot for Explained Variance Ratio Values")
        # plt.xlabel("Concepts(k)")
        # plt.ylabel("Explained Variance Ratio Values")
        # plt.grid(True)
        # plt.axvline(x=knee_point, color='r', linestyle='--', label=f'Knee Point: k={knee_point}')
        # plt.legend()
        # plt.show()

        """Use cumulative_expalined_variance_ratio to find number of components for 80% variance(Rule-of-thumb)"""
        cumsum=0
        for t in range(10,sigma_.shape[0]+10,10):
            cumulative_variance_ratio = self.cumulative_explained_variance_ratio(sigma_, t)
            if (cumulative_variance_ratio>= 0.8):
                num_concepts=t
                cumsum=cumulative_variance_ratio
                print(f"Number of components for 80% variance: {t}")
                break
            
        # CVR=[]
        # for t in range(10,sigma_.shape[0]+10,10):
        #     cumulative_variance_ratio = self.cumulative_explained_variance_ratio(sigma_, t)
        #     CVR.append(cumulative_variance_ratio)
        
        # tn_concepts=np.arange(10,sigma_.shape[0]+10,10)
        # plt.plot(tn_concepts,CVR)
        # plt.title("Cumulative Explained Variance Ratio")
        # plt.xlabel("Number of Concepts")
        # plt.ylabel("Cumulative Explained Variance Ratio")
        # plt.grid(True)
        # plt.axvline(x=num_concepts, color='r', linestyle='--', label=cumsum)
        # plt.legend()
        # plt.show()
        
            

        # Reduce dimensions to num_topics
        U_reduce = U[:, :num_concepts]
        Sigma_reduce = np.diag(Sigma[:num_concepts])
        Vt_reduce = Vt[:num_concepts, :]

 
        lsa_matrix = np.dot(U_reduce, np.dot(Sigma_reduce, Vt_reduce))
        
        return lsa_matrix,U_reduce,Sigma_reduce, Vt_reduce
    
    
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
        qf=self.query_freq(Querylists)

        #Convert documnet dictionary to numpy array
        term_query_matrix=np.zeros((len(self.index.keys()),len(Querylists))) #Make a term document matrix

        for termid in range (len(self.index.keys())):
            term=self.index_list[termid]
            
            for query in range(1,len(Querylists)+1):
                if term in tf_idf_dict[query].keys():
                    term_query_matrix[termid][query-1]=tf_idf_dict[query][term]
        
        return term_query_matrix
    


    def doc_cos_similarity(self,query, documents):
        return cosine_similarity(query.T, documents.T)

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

        self.term_doc_matrix=self.doc_dict_to_numpy(self.docs)
        lsa_doc_term_matrix,U_reduce,Sigma_reduce,Vt_reduce=self.LSA()

        doc_IDs_ordered = []

        start_time = time.time()

        self.term_query_matrix=self.query_dict_to_numpy(queries)
        query_lsa_space=np.dot(self.term_query_matrix.T,U_reduce)
        query_doc_cos_similarity=self.doc_cos_similarity(query_lsa_space.T,np.dot((Sigma_reduce),Vt_reduce)) 

        for query in query_doc_cos_similarity:
            
            sorted_indices = np.argsort(query)[::-1]
            doc_IDs_ordered.append(sorted_indices[:10]+1)

        if(query_doc_cos_similarity.shape[0]==1):  #For custom query
            if(max(query)==0):
                return -1

        end_time = time.time()
        print("Time taken for retrieval",end_time-start_time)
        return doc_IDs_ordered