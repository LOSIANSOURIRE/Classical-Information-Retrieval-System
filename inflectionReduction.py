from util import *

# Add your import statements here


class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""
		ps=PorterStemmer()
		reducedText=[]

		if text!= None:
			for t in text:
				tota=[] #tota is used to store the stemmed words of a particular sentence
				for word in t: #t is a sentence in a particular document
					tota.append(ps.stem(word))
				reducedText.append(tota)


		return reducedText





	# Lemmatization function
	def lemmatize_text(self,text):
		lemmatizer = WordNetLemmatizer()

		reducedText=[]

		for sentence in text:
			# Lemmatize each word based on its part of speech

			lemmatized_words = []
			for word in sentence:
				# # Get the part of speech tag for each word
				pos_tag = nltk.pos_tag([word])[0][1][0].upper()  # First letter of POS tag
				
				# Map POS tags to WordNet tags
				if pos_tag.startswith('J'):
					# Adjective
					pos = wordnet.ADJ
				elif pos_tag.startswith('V'):
					# Verb
					pos = wordnet.VERB
				elif pos_tag.startswith('R'):
					# Adverb
					pos = wordnet.ADV
				else:
					# Default to noun beacasue it is the most common POS and even thought some maynot be nouns, but there is not much harm in assuming them as nouns
					pos = wordnet.NOUN 
				
				# Lemmatize the word using the WordNet lemmatizer
				lemma = lemmatizer.lemmatize(word, pos=pos)
				lemmatized_words.append(lemma)
			
			reducedText.append(lemmatized_words)
		
		return reducedText