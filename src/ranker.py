from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import hazm
import string


class CommentRanker():
	def __init__(self, language):
		self.language=language

		self.lemmatizer_en=WordNetLemmatizer()
		self.stopwords_en=stopwords.words("english")

		self.normalizer=hazm.Normalizer()
		self.word_tokenizer=hazm.WordTokenizer()
		self.lemmatizer_fa=hazm.Lemmatizer()
		self.stopwords_fa=hazm.utils.stopwords_list()


	def tokenize_en(self, comment):
		#Cleaning text from useless characters
		comment=comment.lower()
		comment=re.sub(r"\.", " ", comment)
		comment=re.sub(r"[^a-z0-9 ]", " ", comment)
		
		#Removing stopwords and lemmatizing words
		final_words=[]
		for word in word_tokenize(comment):
			if word not in self.stopwords_en and len(word)>1:
				final_words.append(self.lemmatizer_en.lemmatize(word))
		
		return final_words


	def tokenize_fa(self, comment):
		comment=comment.lower()

		#Removing special and English characters from text
		for i in "!@#$%^&*؛,،»«()/؟?+-\".;:=": comment=comment.replace(i, " ")
		for i in string.ascii_lowercase: comment=comment.replace(i, " ")

		#Normalizing text and splitting it into words
		comment=self.normalizer.normalize(comment)
		comment=comment.replace("آ", "ا")
		words=self.word_tokenizer.tokenize(comment)

		#Lemmatizing words and removing stopwords
		words=map(lambda x: self.lemmatizer_fa.lemmatize(x).split("#")[0], words)
		words=filter(lambda w: w not in self.stopwords_fa and w!="", words)

		return list(words)


	def rank(self, comments):
		#Selecting tokenizer
		if self.language=="en": tokenizer=self.tokenize_en
		elif self.language=="fa": tokenizer=self.tokenize_fa

		#Creating the TF-IDF vectorizer
		vectorizer=TfidfVectorizer(tokenizer=tokenizer)
		result=vectorizer.fit_transform(comments).toarray() #Converting text into TF-IDF vectors
		scores=map(lambda x: sum(x)/len(x), result) #Calculating score for each comment

		#Preparing and sorting the result
		comments=list(zip(comments, scores))
		comments.sort(key=lambda x: x[1], reverse=True)
		return comments