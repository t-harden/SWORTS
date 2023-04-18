# coding = utf-8

from nltk.corpus import stopwords
import string
import nltk
import re

special_title_list = ["FetchandDisplaySpecies", "entreztoKeggImage", "tRNAscan", "RSworkflow slide 32", "Sort 1000 fastest CMEs from a given list according to their pa_width (measured visible angle).", "Joining VOtables with information to execute Sextractor, Galfit and Ellipse.", "FIlter list of strings by regex with an unsuitable regular expression", "getgenesbyspecies.xml"]
def special_title_process(str):
    if str == "FetchandDisplaySpecies":
        return ['Fetch', 'and', 'Display', 'Species']
    elif str == "entreztoKeggImage":
        return ['entre', 'to', 'Kegg', 'Image']
    elif str == "tRNAscan":
        return ['tRNA', 'scan']
    elif str == "RSworkflow slide 32":
        return ['workflow', 'slide']
    elif str == "Sort 1000 fastest CMEs from a given list according to their pa_width (measured visible angle).":
        return ['Sort', 'fastest', 'CMEs', 'from', 'given', 'list', 'according', 'to', 'their', 'pa', 'width', 'measured', 'visible', 'angle']
    elif str == "Joining VOtables with information to execute Sextractor, Galfit and Ellipse.":
        return ['Joining', 'VO' ,'tables', 'with', 'information', 'to', 'execute', 'Sextractor', 'Galfit', 'and', 'Ellipse']
    elif str == "FIlter list of strings by regex with an unsuitable regular expression":
        return ['FIlter', 'list', 'of', 'strings', 'by', 'regex', 'with', 'an', 'unsuitable', 'regular', 'expression']
    elif str == "getgenesbyspecies.xml":
        return ['get', 'gene', 'by', 'species', 'xml']

special_act_list = ["RTwidth_value", "URLencode", "createtmpdir", "ISOtoJD_1"]
def special_act_process(str):
    if str == "RTwidth_value":
        return ['width', 'value']
    elif str == "URLencode":
        return ['URL', 'encode']
    elif str == "createtmpdir":
        return ['create', 'tmp', 'dir']
    elif str == "ISOtoJD_1":
        return ['ISO', 'to', 'JD']

def Pro4Title(str):

    if str in special_title_list:
        return special_title_process(str)
    if str in special_act_list:
        return special_act_process(str)

    for c in str:
        if not c.isalpha():
            str = str.replace(c, " ")

    temp_list = str.split()
    #print(temp_list)
    result_list = []
    for str1 in temp_list:
#         if str1.islower():
#             result_list.extend(wordninja.split(str1))
#         else:
            str1[0].upper()
            new_word = str1[0]
            i = 1
            while (i<len(str1)):
                while ((str1[i].isupper() and str1[i-1].isupper() and i<len(str1)-1) or (str1[i].islower() and i<len(str1)-1)):
                    if (str1[i].isupper() and str1[i-1].isupper() and i<len(str1)-1 and str1[i+1].islower()):
                        break
                    new_word += str1[i]
                    i += 1
                if i==len(str1)-1:
                    new_word += str1[i]
                result_list.append(new_word)
                if i<len(str1):
                    new_word = str1[i]
                    i += 1
                else:
                    break

    new_result_list = []
    for word in result_list:
        word = word.lower()
        if not word in stopwords.words('english'):
            new_result_list.append(word)
    return new_result_list

replacement_patterns = [(r'don\'t', 'do not'), (r'didn\'t', 'did not'), (r'can\'t', 'cannot')]

class RegexpReplacer(object):
   def __init__(self, patterns=replacement_patterns):
      self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

   def replace(self, text):
      s = text
      for (pattern, repl) in self.patterns:
           s = re.sub(pattern, repl, s)
      return s

def Pro4Des(des):


    des = des.lower()

    replacer = RegexpReplacer()
    des = replacer.replace(des)

    remove = str.maketrans('', '', string.punctuation)
    des_without_punctuation = des.translate(remove)

    des_tokens = nltk.word_tokenize(des_without_punctuation)

    des_without_stopwords = [w for w in des_tokens if not w in stopwords.words('english')]

    st = nltk.stem.SnowballStemmer('english')
    des_cleaned_text = [st.stem(ws) for ws in des_without_stopwords]

    return des_cleaned_text

