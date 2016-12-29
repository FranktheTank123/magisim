import math
import json
import logging
import sys
import argparse
import re
import os
# from tqdm import tqdm
from scipy import spatial
import pandas as pd

from stemming.porter2 import stem

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from gensim.models.doc2vec import *

import warnings
warnings.filterwarnings("ignore")

# '/Users/maxlebedev/SWDev/DeckBuilder/phrases', '/Users/maxlebedev/SWDev/DeckBuilder/uniq_phras'
# def construct_unique_phrase_list(phr_path, output):
# 	#code snippet that word2vec's phrase tool and collects all the uniques.
# 	# The goal here is to place these phrases in the cardtext at some point
# 	inp = open(phr_path, 'r')
# 	inp = inp.readlines()
# 	out = open(output, 'w')
# 	set_phr = set()
# 	for line in inp:
# 		words = re.compile(r'[^\s,().]+_[^\s,().]+').findall(line)
# 	for word in words:
# 		if word not in set_phr:
# 			out.write(word+'\n')
# 			set_phr.update({word})

model_foler = "~/dev/magisim/model"
d2v_name = "saved_model"



def get_words(text):
	wlist = re.compile(r'\w+_?\w*').findall(text) # maybe we want to treat all mana costs the same?
	wlist = [value for value in wlist if value not in stop_words]
	return wlist

# loads a set
def load_set():
	with open('/Users/tianyixia/dev/magisim/data/AllCards.json') as data_file: 
		cards = json.load(data_file)
	real_cards = dict() #'real' cards have card text, if a card does not have rules text, it is pointless to care about it
	for cardname, etc in cards.items():
		if 'text' in etc:
			logging.debug('name:: %s',cardname )
			txt = etc['text'].replace(cardname, '~') #replace mentions of the name with ~
			# etc['text'] = stem(txt) <- this only works on word level
			real_cards[cardname] = etc
	return real_cards

def get_cleaned_data():
	"""Cast JSON file into DataFrame, with text cleaned by '~'."""
	return pd.DataFrame.from_dict(load_set(), orient='index')


def process_txt(text):
	# TODO: need to deal with `-`
    return [stem(re.sub('[^A-Za-z0-9{}+]+', '',x)) for x in str(text).lower().split()
                 if x not in stopwords.words('english')]

def get_train_data_stream(df):
    for ( row_ , (card_name, card_)) in enumerate(df.iterrows()):
        if (row_ % 1000 == 0):
            print("Now row", row_, "out of", df.shape[0])
        txt_ = process_txt(card_["text"])
        yield LabeledSentence(txt_, [card_name.lower()])

def get_train_data():
	return list(get_train_data_stream(get_cleaned_data()))

def get_doc2vec():
	target_model_path = os.path.join(model_foler, d2v_name)

	try:
		model = Doc2Vec.load(target_model_path)
		logging.info("Doc2Vec model {} load successfully.".format(d2v_name))
	except:
		logging.info("Doc2Vec model not found. Will train one.")
		model = Doc2Vec(alpha=0.025, min_alpha=0.025, workers=3)
		train_data = get_train_data()
		model.build_vocab(train_data)
		for epoch in range(10):
			model.train(train_data)
			model.alpha -= 0.002  # decrease the learning rate
			model.min_alpha = model.alpha  # fix the learning rate, no decay
			logging.info("epoch {} done".format(epoch+1))
		model.save(target_model_path)
		logging.info("New Doc2Vec model saved to {}".format(target_model_path))
	return model

def get_similar_card_doc2vec(card, topn=10, model=None):
	if(model == None):
		print("get model")
		model = get_doc2vec()
	# return [x for x in model.docvecs.most_similar(card.lower(), topn=topn)]
	return model.docvecs.most_similar(card.lower(), topn=topn)

def print_top_n_d2v(card, topn):
	logging.info("\n\n************************************* Doc2Vec Results for {} *************************************\n".format(card))

	for ind, entry in enumerate(get_similar_card_doc2vec(card, topn)):
		logging.info('match: %s | score: %f', entry[0], entry[1])

	logging.info("\n************************************* Doc2Vec Results End *************************************\n\n".format(card))

def sklearn(cards, card):
	txtlst = list()
	names = list()
	for i, (c,v) in enumerate(cards.items()):
		if card.lower() == c.lower():
			card = c
		txtlst.append(v['text'])
		names.append(c)
	crdind = names.index(card)

	tfidf_vectorizer = TfidfVectorizer(token_pattern='\w+')
	tfidf_matrix = tfidf_vectorizer.fit_transform(txtlst)

	csim = cosine_similarity(tfidf_matrix[crdind:crdind+1], tfidf_matrix)[0]

	simdict = dict()
	for i in range(len(names)):
		simdict[names[i]] = csim[i]
	return simdict


def print_top_n_tfidf(card, cards, n):
	logging.info("\n\n************************************* TFIDF Results for {} *************************************\n".format(card))
	#separate function
	for ind, entry in enumerate(sorted(cards, key=cards.get, reverse=True)):
		if ind > n:
			logging.info("\n************************************* TFIDF Results End *************************************\n\n".format(card))
			return
		logging.info('match: %s | score: %f', entry, cards[entry])
	


def cosine_sim(keys, dct1, dct2):
	lst1 = list()
	lst2 = list()
	for k in keys:
		lst1.append(dct1[k])
		lst2.append(dct2[k])
	logging.debug('lst2 %s', str(lst2))
	res = 1 - spatial.distance.cosine(lst1, lst2)
	return res


# main loop
def repl(cards, num_res):
	while True:
		card = input("Please enter a card name (type 'exit' to break): ")
		engine = input('Please enter a card search engine: "d" for Doc2Vec, "t" for TFIDF: ')
		if not card:
			continue
		if engine not in ["d", "t"]:
			continue
		if card == 'exit':
			break
		else:
			try:
				if engine == "t":
					sim_dict = sklearn(cards, card.strip())
					print_top_n_tfidf(card, sim_dict, num_res)
				else: # engine == "d"
					print_top_n_d2v(card, num_res)
			except Exception as e:
				logging.info(e)
				continue


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", '--verbose', action="store_true", help="Verbose output")
	parser.add_argument("-n", '--num', action="store", help="number of results to show")
	args = parser.parse_args()

	lvl = logging.INFO
	if args.verbose:
		lvl = logging.DEBUG
	logging.basicConfig(stream=sys.stderr, level=lvl)
	cards = dict()
	cards.update(load_set()) #mtgset being AllCards
	num_res = 10
	if args.num:
		num_res = int(args.num)
	repl(cards, num_res)
	
#TODO long/complicated keywords are weighted too highly
#TODO 2 card mode where we get deets on the matching. maybe
#TODO add other similarity metrics
