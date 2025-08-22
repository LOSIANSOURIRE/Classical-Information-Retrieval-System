from util import *

"""Function to construct the vocabulary of the corpus"""
# class   Vocabulary:
#     def vocabulary(self, inverted_index):
#         """
#         Vocabulary

#         Parameters
#         ----------
#         arg1 : A dictionary containing the inverted index

#         Prints
#         -------
#         Vocabulary of the corpus
            
#         """
#         vec={}
#         bigram=[a+b for a in alphabets for b in alphabets]
#         for t in inverted_index.keys():
#             if t.isalpha():
#                 vec[t]={}
#             for i in bigram:
#                 if t.isalpha() and i in t:
#                     if i in vec[t]:
#                         vec[t][i]+=1
#                         print(vec[t][i])
#                     else:
#                         vec[t][i]=1
#         print(vec)

class CandidateSelection:

    def bigrams(self, word,args):
        """
        Candidate Selection

        Parameters
        ----------
        arg1 : string
            A word for which bigrams are to be generated

        Returns
        -------
        list
            A set of bigrams for the word
        """
        bigram=[a+b for a in alphabets for b in alphabets]
        word=word.lower()
        bigram_list=set()
        for i in bigram:
         if i in word:
            bigram_list.add(i)
        json.dump(bigram_list, open(args + "bigram_list_vocabulary.txt", 'w'))
     
        return bigram_list
    
    def jaccard_coefficient(self, bigram_list1, bigram_typo):
        """
        Jaccard Coefficient

        Parameters
        ----------
        arg1 : list
            A dictionary with the word as the key and a set of bigrams as the value
        arg2 : list
            A dictionary with the word as the key and a set of bigrams as the value

        Returns
        -------
        float
            The Jaccard Coefficient between the two sets of bigrams
        """
        intersection = len(bigram_list1.intersection(bigram_typo))
        union = len(bigram_list1.union(bigram_typo))
        jaccard_coefficient = intersection / union
        
        return jaccard_coefficient
