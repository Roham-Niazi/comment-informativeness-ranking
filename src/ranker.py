from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize, download
import re
import hazm
import string

#Downloading nltk dependencies
download("stopwords")
download("wordnet")
download("punkt")
download("punkt_tab")


class CommentRanker():
	def __init__(self, language):
		self.language=language

		self.lemmatizer_en=WordNetLemmatizer()
		self.stopwords_en=stopwords.words("english")

		self.normalizer=hazm.Normalizer()
		self.word_tokenizer=hazm.WordTokenizer()
		self.lemmatizer_fa=hazm.Lemmatizer()
		self.stopwords_fa=hazm.utils.stopwords_list()


	def preprocess_en(self, comment):
		#Cleaning text from useless characters
		comment=comment.lower()
		comment=re.sub(r"\.", " ", comment)
		comment=re.sub(r"[^a-z0-9 ]", " ", comment)
		
		#Removing stopwords and lemmatizing words
		final_words=[]
		for word in word_tokenize(comment):
			if word not in self.stopwords_en and len(word)>1:
				final_words.append(self.lemmatizer_en.lemmatize(word))
		
		return " ".join(final_words)

	def remove_chars(self, text, chars):
		for char in chars: text=text.replace(char, " ")
		return text

	def preprocess_fa(self, comment):
		comment=comment.lower()

		#Removing punctuation marks and English characters from text
		persian_punctuation="،؛؟٪«»‹›–—…٬٫"
		comment=self.remove_chars(comment, string.punctuation+persian_punctuation)
		comment=re.sub(r"[a-zA-Z]", " ", comment)

		#Normalizing text and splitting it into words
		comment=self.normalizer.normalize(comment).replace("آ", "ا")
		comment=self.remove_chars(comment, "۰۱۲۳۴۵۶۷۸۹\u200c")

		words=self.word_tokenizer.tokenize(comment)


		#Lemmatizing words and removing stopwords
		words=map(
			lambda x: self.lemmatizer_fa.lemmatize(x).split("#")[0],
			words
		)
		words=filter(
			lambda w: w not in self.stopwords_fa and w!="",
			words
		)

		return " ".join(list(words))


	def rank(self, comments):
		#Cleaning comments
		if self.language=="en":
			cleaned_comments=list(map(self.preprocess_en, comments))
		elif self.language=="fa":
			cleaned_comments=list(map(self.preprocess_fa, comments))

		#Creating the TF-IDF vectorizer
		vectorizer=TfidfVectorizer()
		result=vectorizer.fit_transform(cleaned_comments).toarray() #Converting text into TF-IDF vectors
		scores=map(lambda x: sum(x)/len(x), result) #Calculating score for each comment

		#Preparing and sorting the result
		comments=list(zip(comments, scores))
		comments.sort(key=lambda x: x[1], reverse=True)
		return comments