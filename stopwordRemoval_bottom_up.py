from util import *

class StopwordRemoval_bottom_up():

	def fromList(self, text,idf):
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
		

		#Took threshold to be ln(3) assuming that those words which are present in atleast 1/3rd of the corpus are stopwords
		sw =[i for i in idf.keys() if idf[i]<math.log(2)]
		stopwordRemovedText=[]
		
		for t in text:
			tota=[]
			for word in t:
				if word not in sw:
					tota.append(word)
			stopwordRemovedText.append(tota)
			print(stopwordRemovedText)

		

		return stopwordRemovedText




	