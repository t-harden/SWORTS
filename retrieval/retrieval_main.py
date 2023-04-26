# coding = utf-8

import pickle
import preselect
import ranking
import time
import openpyxl
import math

def read_dict():

    with open('../data/myExperiment/dict_BW_myExperiment.pkl', 'rb') as f:
        dict_BW = pickle.load(f)
    with open('../data/myExperiment/dict_GED_myExperiment.pkl', 'rb') as f:
        dict_GED = pickle.load(f)
    with open('../data/myExperiment/dict_MS_myExperiment.pkl', 'rb') as f:
        dict_MS = pickle.load(f)
    with open('../data/myExperiment/dict_PS_myExperiment.pkl', 'rb') as f:
        dict_PS = pickle.load(f)

    return dict_BW, dict_GED, dict_MS, dict_PS


if __name__ == '__main__':


    dict_BW, dict_GED, dict_MS, dict_PS = read_dict()
    query_list = [202, 124, 217, 773, 1014] # Define your query list here

    k = 100  # N'
    kk = 10  # N
    "Preliminary Selection Step"
    tfidf_candidate = preselect.work(query_list, k, dict_BW, dict_PS)
    print("tfidf_candidate：", tfidf_candidate)

    "Ranking Step"
    tfidf_sort = ranking.work(tfidf_candidate, kk)
    print("tfidf_sort：", tfidf_sort)




