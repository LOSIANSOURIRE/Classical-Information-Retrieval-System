from util import *

class SentenceSegmentation():

	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		segmentedText_unfiltered = tokenizer.tokenize(text)
		segmentedText = list(filter(None, segmentedText_unfiltered)) #removed the empty strings
		
		return segmentedText