from util import *

class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		
		sw =stopwords.words('english') #sw is the list of stopwords from the NLTK library
		stopwordRemovedText=[]

		for t in text:
			tota=[] #tota is used to store the stemmed words of a particular sentence
			for word in t:#t is a sentence in a particular document
				if word not in sw:
					tota.append(word)

			stopwordRemovedText.append(tota)

		

		return stopwordRemovedText




	