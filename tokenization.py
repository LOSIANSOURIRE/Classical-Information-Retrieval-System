from util import *

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""


		tokenizedText=[]
		for t in text:
			tokenizedText_unfiltered = re.split(r"[\s]", t)
			tota=list(filter(None, tokenizedText_unfiltered))#removed the empty strings
			tokenizedText.append(tota)

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizer=TreebankWordTokenizer()
		tokenizedText=[]
		pattern = r'[^\w\s]' #removing all the special characters from the text

		for t in text:
			tota_unfiltered =tokenizer.tokenize(re.sub(pattern,' ',t))
			tota=list(filter(None, tota_unfiltered))#removed the empty strings
			tota=[word.lower() for word in tota]
			tokenizedText.append(tota)

		return tokenizedText