# coding = utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import Preprocess4Title_Des

def get_tfidf(dict_BW):
    tag_tfidf = {}
    document = []
    for i,k in enumerate(dict_BW.keys()):
        tag_tfidf[k] = i
        title_word = Preprocess4Title_Des.Pro4Title(dict_BW[k][0])
        des_word = Preprocess4Title_Des.Pro4Des(dict_BW[k][1])
        word_list = title_word + des_word
        s_cleaned_text = ''
        for word in word_list:
            s_cleaned_text = s_cleaned_text + ' ' + word
        document.append(s_cleaned_text)
    tfidf = TfidfVectorizer().fit_transform(document)
    return tfidf, tag_tfidf

def sim(key1, key2, tfidf, tag_tfidf):

    cosine_similarity = float(linear_kernel(tfidf[tag_tfidf[key1]], tfidf[tag_tfidf[key2]]))
    return cosine_similarity
