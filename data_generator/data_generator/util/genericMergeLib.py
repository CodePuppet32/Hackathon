#
# genericMergeLib.py
#
# $Id: genericMergeLib.py,v 1.9 2014/08/21 13:10:35 manoj Exp $
import re, sys
from .lang_utils import latin_ascii_xlate_ext, latin_ascii_xlate

class myfile():
    def __init__(self,name,mode, encoding = 'utf8'):
        self.fp = open(name, mode, encoding=encoding)
        self._encoding = encoding

    def close(self):
        self.fp.close()

    def mywrite(self,string):
        self.fp.write(string)

    def write_heading(self, string):
        underline = '='*len(string)
        final_string = "%s\n%s\n"%(string.rstrip('\n'), underline)
        self.mywrite(final_string)

def convert_latin1_ascii(s, ignore_extended=True, encoding='utf8'):
    global latin_ascii_xlate, latin_ascii_xlate_ext
    try:
        if not isinstance(s, str):
            s = s.decode(encoding)
    except Exception as e:
        return ''
    if ignore_extended:
        xtable = latin_ascii_xlate
    else:
        xtable = latin_ascii_xlate_ext
    return s.translate(xtable)

def keep_only_alpha_num(str1):
    str2=""
    for c in str1:
        if re.match('[a-z]|[0-9]', c.lower()) == None:
            continue
        str2=str2+c.lower()
    return str2


def remove_non_alphabets(str1):
    str2=""
    for c in str1:
        if c.lower() in "'\",-;\r":
            continue
        str2=str2+c.lower()

    return str2

def namelist_compare_fuzzy( tv_cast, db_cast):
    ii=0
    jj=0
    cast_cnt = 0
    while ii < len(tv_cast):
        jj=0
        while jj < len(db_cast):
            t_cast = tv_cast[ii]
            d_cast = db_cast[jj]
            #t_cast = keep_only_alpha_num(t_cast)
            #t_cast = remove_non_alphabets(t_cast)
            t_cast =  t_cast.split("{")[0]
            t_cast = t_cast.replace(".", "").lower()
            t_cast = t_cast.replace(",", "")
            #t_cast = t_cast.replace("{"," ")
            #t_cast = t_cast.replace("}"," ")
            #d_cast = keep_only_alpha_num(d_cast)
            #d_cast = remove_non_alphabets(d_cast)
            d_cast =  d_cast.split("{")[0]
            d_cast = d_cast.replace(".", "").lower()
            d_cast = d_cast.replace(",", "")
            #d_cast = d_cast.replace("{"," ")
            #d_cast = d_cast.replace("}"," ")
            dl = d_cast.split(" ")
            tl = t_cast.split(" ")
            cnt = 0
            iii=0
            while iii<len(tl):
                jjj=0
                if tl[iii]=="" or not tl[iii][0].isalpha() or len(tl[iii])==1:
                    iii=iii+1
                    continue
                while jjj<len(dl):
                    if tl[iii]==dl[jjj]:
                        cnt=cnt+1
                    jjj=jjj+1
                iii=iii+1
            if cnt > 1:
                cast_cnt=cast_cnt+1
            jj=jj+1
        ii=ii+1
    return cast_cnt


def compress_spaces(string):
    newstr=""
    i=0
    lenstr = len(string)
    while i < lenstr:
        if string[i]==" " or string[i]=='\t':
            if len(newstr)>0:
                newstr = newstr+" "
            i=i+1
            while i < lenstr and (string[i]==" " or string[i]=='\t'):
                i=i+1
            continue
        else:
            newstr = newstr+string[i]
            i=i+1

    if len(newstr) and newstr[len(newstr)-1]==" ":
        newstr=newstr[:len(newstr)-1]
    #elif len(newstr)==0:
        #print string.encode("latin-1")

    return newstr

def is_word_decstr(word):
    decstr = "0123456789:-"

    if len(word)==0:
        return 0

    if word[0]=="0":
        return 0
    if len(word) > 2:
        return 0
        for i in word:
            if i not in decstr:
                return 0
        return 1

def is_word_romstr(word):
    romstr = "IVX:-"
    for i in word:
        if i.lower() not in romstr.lower():
            return 0
    if len(word)>1 or (len(word)==1 and word.lower() != "i"):
        return 1
    if word.lower() == 'i':
        return 2
    return 0

def is_word_number(word):
    numberlist=["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",\
            "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]

    if len(word) < 1:
        return 0
    if word[-1] == ":":
        word=word[:-1]

        if word.lower() in numberlist:
            return 1
        else:
            return 0

def strip_off_leading_article(s1):
    single_l = ["a ", "l "]
    two_l = ["le ", "la ", "an ", "de ", "un ", "os ", "el ", "en "] # "as" is removed  because it causes problems.
    three_l = ["les ", "las ", "det ", "une ", "the ", "los "]
    if len(s1) < 1:
        return s1
    if s1[:2] in single_l:
        return s1[2:]
    if s1[:3] in two_l:
        return s1[3:]
    if s1[:4] in three_l:
        return s1[4:]
    return s1


def strip_off_leading_english_article(s1):
    single_l = ["a "]
    two_l = ["an "] # "as" is removed  because it causes problems.
    three_l = ["the "]
    if len(s1) < 1:
        return s1
    if s1[:2] in single_l:
        return s1[2:]
    if s1[:3] in two_l:
        return s1[3:]
    if s1[:4] in three_l:
        return s1[4:]
    return s1

def rotate_leading_article(s1):
    single_l = ["a ", "l "]
    two_l = ["le ", "la ", "an ", "de ", "un ", "os ", "el ", "en "] # "as" is removed  because it causes problems.
    three_l = ["les ", "las ", "det ", "une ", "the ", "los "]
    if len(s1) < 1:
        return s1
    if s1[:2] in single_l:
        return s1[2:]+", "+s1[:1]
    if s1[:3] in two_l:
        return s1[3:]+", "+s1[:2]
    if s1[:4] in three_l:
        return s1[4:]+", "+s1[:3]
    return s1


def reformatstring(str1):
    single_l = [" a", " l"]
    two_l = [" le", " la", " an", " de", " un", " os", " el", " en"] # "as" is removed  because it causes problems.
    three_l = [" les", " las", " det", " une", " the", " los"]


    strlen = len(str1)
    i = strlen-1
    while i >= 0:
        if str1[i] == " " or str1[i]=='\t':
            i=i-1
        else:
            break
    ## No trailing spaces now
    newstr = str1[0:i+1]
    r1 = newstr[-4:].lower()
    if r1 in three_l:
        article = r1[1:]
        newstr = article[0].upper()+article[1:]+" "+ newstr[:-4]
        if newstr[len(newstr)-1]==",":
            newstr=newstr[0:len(newstr)-1]
        return newstr
    r1 = newstr[-3:].lower()
    if r1 in two_l:
        article = r1[1:]
        newstr = article[0].upper()+article[1:]+" "+ newstr[:-3]
        if newstr[len(newstr)-1]==",":
            newstr=newstr[0:len(newstr)-1]
        return newstr
    r1 = newstr[-2:].lower()
    if r1 in single_l:
        article = r1[1:]
        newstr = article[0].upper()+" "+ newstr[:-2]
        if newstr[len(newstr)-1]==",":
            newstr=newstr[0:len(newstr)-1]
        return newstr

    if len(newstr) and newstr[len(newstr)-1]==",":
        newstr=newstr[0:len(newstr)-1]
    return newstr

def only_dashes_or_colons(s):
    ii=0
    for c in s:
        if c in ":-":
            ii=ii+1
            continue
        else:
            return 0
    return ii

def is_word_partword(w2, next):
    if w2.lower() not in ["part", "vol.", "vol", "volume", "episode", "no."]:
        return 0
    if is_word_decstr(next) or is_word_romstr(next) or is_word_number(next):
        return 1
    return 0

def pad_colons_dashes(s):
    if len(s) == 0:
        return s
    out = s[0]
    ii = 1
    len_s = len(s)
    while ii < len_s:
        if (s[ii] == ":" or s[ii] == "-") and (s[ii-1] != " " and s[ii-1] not in "0123456789"):
            out = out+" "+s[ii]
        else:
            out = out+s[ii]
        ii=ii+1

    return out

def numeric_partstr(w2, flist, ii):
    adv = 0
    if is_word_decstr(w2) or is_word_romstr(w2):
        if (ii == len(flist)-1) or w2[-1] == ":" or w2[-1] == "-":
            adv = 1
        else:
            if only_dashes_or_colons(flist[ii+1]):
                adv = 2
    return adv

def part_meaning(sepwordlist):
    comsepstr = ""
    for w in sepwordlist:
        w1 = w.replace(":", "")
        w1 = w1.replace("-", "")
        if is_word_decstr(w1) or is_word_romstr(w1) or (len(w1)==1 and w1.lower() in "ivx"):
            if len(comsepstr):
                comsepstr = comsepstr+","+w1
            else:
                comsepstr = w1

    return comsepstr

def string_tokenizer(full_ascii):
    full_ascii = pad_colons_dashes(full_ascii)
    flist = [tok for tok in full_ascii.split(" ") if tok]
    outlist =[]
    sep_start_index = -1
    word_start_index = 0

    #First word is not a separator
    ii=1
    while ii < len(flist):
        w = flist[ii]
        w1 = w.replace("'", "")
        w2 = w1.replace("\"", "")
        #case 1: single separator character : "-" or ":"
        if only_dashes_or_colons(w2):
            if sep_start_index == -1:
                #Start of a separator substr
                sep_start_index = ii
                mr_title = " "+" ".join(flist[word_start_index:ii])
                outlist = outlist + [[1, mr_title]]
                word_start_index = -1
            ii=ii+1
            continue

        #case 2: decimal number or roman number
        adv = numeric_partstr(w2, flist, ii)
        if adv:
            if sep_start_index == -1:
                #Start of a separator substr
                sep_start_index = ii
                mr_title = " "+" ".join(flist[word_start_index:ii])
                outlist = outlist + [[1, mr_title]]
                word_start_index = -1
            ii=ii+adv
            continue

        #case 3: one of the words part or episode or volume or vol. or no.
        if ii < len(flist)-1 and is_word_partword(w2, flist[ii+1]):
            if sep_start_index == -1:
                #Start of a separator substr
                sep_start_index = ii
                mr_title = " "+" ".join(flist[word_start_index:ii])
                outlist = outlist + [[1, mr_title]]
                word_start_index = -1
            ii=ii+2
            continue

        #non separator
        if sep_start_index != -1:
            #End of a separator substr
            sepstr = " ".join(flist[sep_start_index:ii])+"\t\t"+part_meaning(flist[sep_start_index:ii])
            outlist = outlist + [[2, sepstr]]
            sep_start_index  = -1

        if word_start_index == -1:
            word_start_index = ii

        ii = ii+1
        continue

    if sep_start_index != -1:
        #Start of a separator substr
        sepstr = " ".join(flist[sep_start_index:])+"\t\t"+part_meaning(flist[sep_start_index:ii])
        outlist = outlist + [[2, sepstr]]

    if word_start_index!=-1:
        mr_title = " "+" ".join(flist[word_start_index:])
        outlist = outlist + [[1, mr_title]]

    # Here we adjust the strings emnating out of the list to handle the case where a space precedes the : or -
    # This is due to the fact that we introduced these spaces in the beginning of the procedure.
    for tempItem in outlist:
        if tempItem[0] == 1:
            while (tempItem[1].find(': ') != -1) or (tempItem[1].find(' :') != -1) or (tempItem[1].find('- ') != -1) or (tempItem[1].find(' -') != -1):
                tempItem[1] = tempItem[1].replace(': ', ':')
                tempItem[1] = tempItem[1].replace(' :', ':')
                tempItem[1] = tempItem[1].replace('- ', '-')
                tempItem[1] = tempItem[1].replace(' -', '-')
    return outlist


post_t_l1=["1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "9 ", "0 ", "iii", "ii", "part i", "vol i", "vol. i", "volume i", "episode i", "part iv", "vol iv", "vol. iv", "volume iv", "episode iv"]
post_t_l2=["one ", "two ", "three ", "four ", "five ", "six ", "seven ", "eight ", "nine ", "zero ", "three", "two", "part one", "vol one", "vol. one", "volume one", "episode one", "part four", "vol four", "vol. four", "volume four", "episode four"]

def post_tokenizer_replace_spl_chars(str2):
#    str2=keep_only_alpha_num(str2)
    #str2=remove_non_alphabets(str2)
    jj=0
    while jj<22:
        if post_t_l1[jj] in str2:
            str2=str2.replace(post_t_l1[jj], post_t_l2[jj])
        jj=jj+1
    str2=str2.replace("vol.", "vol")
    if str2[-2:]=="iv":
        str2=str2[:-2]+"four"
    str2=keep_only_alpha_num(str2)
    #str2=compress_spaces(str2)
    return str2

def pre_tokenizer_replace_spl_chars(str1):
    #print str1
    str1=str1.replace("&amp;", "and ")
    str3=str1.replace("$", "dollar ")
    str3=str3.replace("&", "and ")
    str3=compress_spaces(str3)
    return str3

def get_list_of_hashstrs(str1):
    str1 = str1.lower()
    reft = str1.replace("'", "")
    reft = reft.replace("\"", "")
    reft = pre_tokenizer_replace_spl_chars(reft)
    reft = reformatstring(reft)
    l3 = reft.lower()
    ct = convert_latin1_ascii(l3)
    parts = string_tokenizer(ct)
    jj=0
    rstr1 = []
    while jj < len(parts):
        p=parts[jj]
        if p[0]==1:
            cstr = p[1]
            m=re.search('\([^\)]*\)', cstr)
            if m != None:
                if m.end(0)<len(cstr):
                    cstr1 = (cstr[:m.start(0)]+cstr[m.end(0):]).strip()
                    cstr2 = cstr[m.start(0):m.end(0)].strip()
                else:
                    cstr1=cstr[:m.start(0)].strip()
                    cstr2=cstr[m.start(0):m.end(0)].strip()
                if len(cstr1):
                    rstr1=rstr1+[cstr1]
                if len(cstr2):
                    rstr1=rstr1+[cstr2]
            else:
                rstr1= rstr1+[cstr.strip()]
        jj=jj+1
    rstr2=[]
    concatstr = ""
    for s in rstr1:
        s1=post_tokenizer_replace_spl_chars(s).strip()
        if len(s1)==0:
            continue
        rstr2=rstr2+[s1]
        if len(concatstr):
            concatstr = concatstr.strip()+s1
        else:
            concatstr=s1
    if len(rstr2)>1:
        rstr2=rstr2+[concatstr]
    #Only if there arent any partstrings do we introduce aka titles without
    #leading articles --
    #to be tweaked to cover common Euro languages

    #if len(rstr1)==1 :
    #    newstr = strip_off_leading_article(rstr1[0])
    #    if newstr != rstr1[0]:
    #        newstr = post_tokenizer_replace_spl_chars(newstr).strip()
    #        if len(newstr):
    #        rstr2=rstr2+[newstr]

    for e in rstr1:
        n_e = strip_off_leading_article(e)
        if e != n_e:
            n2 = post_tokenizer_replace_spl_chars(n_e).strip()
            rstr2=rstr2+[n2]

    return rstr2

def nonAlphaSplitter(ipStr):
    """splits on delimiter , where delimit = anything other than a-z"""
    finalList={} # using dict - so that can return only unique token
    token=''
    for ch in ipStr.lower():
        if  ch not in 'abcdefghijklmnopqrstuvwxyz' : # encountered a delim..split now
            if len(token)>0: # else its just a consecutive delimiter
                finalList[token]=1

            token=''
        else:
            token+= ch

    if len(token)>0: #handling the last -leftover token
        finalList[token]=1
    return list(finalList.keys())


def evaluateCosineSimilarity(list1, list2): #assumes lists contain unique tokens ( like sets)
    interSectionCount=0

    if len(list1)<=0 or len(list2)<=0 :
        return 0

        for token in list1 :
            if token in list2 :
                interSectionCount+= 1

    unionCount = len(list1) + len(list2) -interSectionCount
    return 1.0*interSectionCount/unionCount


def evaluateOverlapScore(list1, list2): #assumes lists contain unique tokens ( like sets)
    interSectionCount=0

    if len(list1)<=0 or len(list2)<=0 :
        return 0

        for token in list1 :
            if token in list2 :
                interSectionCount+= 1

    if len(list1) < len(list2) :
        return 1.0*interSectionCount/len(list1)
    else:
        return 1.0*interSectionCount/len(list2)


def filterWords(origList, removeList):
    retList=[]
    for token in origList:
        if token not in removeList:
            retList.append(token)
    return retList


def uniquifyTokenList(list):
    # given a list , it returns a copy of the given list with no duplicates
    retList = {}
    for token in list:
        retList[token]=1

    return list(retList.keys())


def nGrammer(ipStr,n=3): # return a list of n-grams of the str . default n==3
    str1= ipStr.lower()
    retHash={}

    tok=[]
    for ch in str1[:n]:
        tok.append(ch)
    retHash[ ''.join(tok) ]=1
    for ch in str1[n:]:
        tok.pop(0)
        tok.append(ch)
        retHash[ ''.join(tok) ]=1


    """
    #initial n-grams (before n'th pos in str1)
    for ch in str1[:n]:
        tok.append(ch)
        retHash[ ''.join(tok) ]=1

    #in b/w n-grams
    for ch in str1[n:] :
        tok.pop(0) # pop
        tok.append(ch)
        retHash[''.join(tok) ]=1

    # the last few characters
    for idx in range(len(tok)) :
        retHash[''.join(tok[idx:]) ]=1

    """
    return list(retHash.keys())

if __name__ == '__main__' : # just used for testing
    l1=    nGrammer(sys.argv[1], 2)
    l2=    nGrammer(sys.argv[2], 2)
    print(("lists:", l1, l2))
    print(("score= ", evaluateCosineSimilarity(l1, l2)))
