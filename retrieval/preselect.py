# coding = utf-8


import utils.tfidf as tfidf
import time
import numpy as np



def work(query_list, k, dict_BW, dict_PS):

    dict_Path_BW = dict_BW
    for k in dict_BW.keys():
        newstr = dict_BW[k][0]
        for temp_list in dict_PS[k]:
            for s in temp_list:
              newstr = newstr + ' ' + s
        dict_Path_BW[k][0] = newstr

    tfidf_model, tag_tfidf = tfidf.get_tfidf(dict_Path_BW)

    tfidf_candidate = []
    for x in range(len(query_list)):
        i = query_list[x]
        tfidf_list_temp = []
        for j in dict_BW.keys():
            tup = (tfidf.sim(i, j, tfidf_model, tag_tfidf), i, j)
            tfidf_list_temp.append(tup)

        tfidf_list_temp = sorted(tfidf_list_temp, reverse=True)
        tfidf_candidate.append(tfidf_list_temp[1: k + 1])

    return tfidf_candidate




