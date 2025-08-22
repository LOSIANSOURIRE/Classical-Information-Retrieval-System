# Add your import statements here
import nltk
# import torch
nltk.download('punkt')
from nltk.tokenize import PunktSentenceTokenizer,TreebankWordTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
import re
import math
import string
alphabets=string.ascii_lowercase
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.metrics import edit_distance
nltk.download('wordnet')
import time
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
import argparse
import json
from sys import version_info
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load('en_core_web_sm')
import nltk
nltk.download('punkt')
from nltk.tokenize import TreebankWordTokenizer
import re
from scipy import stats
import seaborn as sns

# Add any utility functions here