#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# $Id: StringUtil.py,v 1.77 2017/09/18 08:45:26 lijin.chungapalli Exp $

import re, sys, os
import string
import unittest

from . import genericMergeLib
from .lang_utils import read_json_file, LANG_MAP

try:
    dummy = set() #Python2.4 >=
except:          #Python2.4 <
    import sets
    set = sets.Set

singleQuoteRE = re.compile( r'(\')(?=(s|t|re)(\s|$)?)')
bracketRE = re.compile( r'(\([^\(\)]*\))')
commonCharsRE = re.compile(r'[A-Za-z0-9 ]')
trialRE = re.compile( r'\w+')
SPECIAL_CHAR_RE = re.compile(r'[()\[\]./@\'"-]')
DIGIT_SPECIAL_CHAR_RE = re.compile(r'(?<=[^0-9])(,|:)')
ampercentRe = re.compile(r'\s(&|&amp;)\s')
stripApostsRe = re.compile(r"([^0-9])[`']s ")
stripApostOnlyRe = re.compile(r"[`']")
backslashWRe = re.compile(r'\sw/\s')
COMMON_AND_RE = re.compile(r'\sand\s')
compressSpaceRe = re.compile(r'\s+')
COMMON_LEN = 15
BRACKET_STRING_RE = re.compile(r'(?P<text>[^(]*)\((?P<br_text>[^)]*)\)$', re.I|re.U)
#
capitalize_pat = re.compile(r'\b[A-Z][^A-Z]')
vtv_alphanumeric = string.ascii_letters + string.digits
vtv_ascii_table            = "".join([chr(i) for i in range(256)])
vtv_non_printable_table    =  "".join([_x for _x in vtv_ascii_table if _x in set(string.printable)])
vtv_non_alphanumeric_table = "".join([_x for _x in vtv_ascii_table if _x in set(vtv_alphanumeric)])
vtv_non_space_table        = "".join([_x for _x in vtv_ascii_table if _x in set(string.whitespace)])

vtv_non_alphanumeric_to_space_table = "".join([chr(x) if x not in set(vtv_non_alphanumeric_table) else ' ' for x in range(256)])
vtv_non_alphanumeric_to_space_and_lower_table = vtv_non_alphanumeric_to_space_table.lower()

spam_words = set(["iran", "war", "news", "topless", "bottomless", "tits", "boobs", "cunt", "pussy", "titney", "cats", "dogs", "mice", "rats", "crocodile", "hunter", "australia", "japan", "tokyo", "america", "york", "old", "9/11", "trade", "world", "center", "grand", "theft", "auto", "halo", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99 " "1000", "10", "100", "50", "clitoris", "sexual", "boxing", "basketball", "baseball", "soccer", "tennis", "sports", "NBA", "NFL", "ESPN", "horror", "comedy", "drama", "romance", "ninja", "shinobi", "cosplay", "hentai", "titties", "boobies", "breasts", "bouncing", "beach", "bitches", "bikini", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "hi", "bye", "rachel", "amanda", "sarah", "tony", "dad", "mom", "aunt", "uncle", "grandma", "grandpa", "college", "production", "drums", "singing", "american", "idol", "lost", "heroes", "discover", "animal", "planet", "furry", "yiff", "tupac", "rap", "rock", "jazz", "punk", "emo", "fall", "out", "boy", "in", "door", "panic", "at", "the", "disco", "sugar", "sweet", "pop", "piss", "christina", "anal", "blowjob", "bj", "family", "guy", "simpsons", "fox", "news", "online", "outkast", "birds", "bees", "christian", "athiest", "movies", "demonstration", "japanese", "eyes", "world", "record", "book", "yugioh", "detective", "conan", "case", "closed", "combo", "glitch", "maple", "story", "gunbound", "population", "global", "warming", "al", "gore", "politicians", "political", "white", "house", "jesus", "god", "war", "bomb", "nuke", "osama", "saddam", "k.fed", "fed", "chicken", "popozao", "po", "po", "zao", "concert", "guitar", "hero", "music", "cds", "cd",
        "method", "man", "men", "group", "jay-z", "ashanti", "myspace", "google", "aliens", "UFO", "conspiracy", "government", "denial", "explosion", "mission", "impossible", "tom", "cruise", "scientology", "re:", "reply", "jefferson", "lincoln", "logs", "ren", "stimpy", "chinese", "korean", "electric", "computer", "stopmotion", "stop", "motion", "go", "traffic", "cars", "chevy", "ford", "california", "south", "dakota", "texas", "florida", "michigan", "canada", "mexico", "mexican", "spanish", "translated", "games", "parties", "party", "drunk", "makeout", "kissing", "religious", "scandal", "truth", "real", "fake", "reality", "tv", "television", "game", "show", "blooper", "accident", "tag", "team", "wrestling", "WWE", "WWF", "NWO", "hulk", "hogan", "pamela", "anderson", "king", "hill", "david", "letterman", "jay", "leno", "late", "nite", "show", "dance", "revolution", "bemani", "drummania", "beatmania", "maniac", "70s", "80s", "90s", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "70's", "80's", "90's", "60's", "60s", "50's", "50s", "40's", "40", "pearl", "harbor", "tsunami", "enron", "money", "cash", "bling", "seal", "cat", "dog", "bird", "parakeet", "editing", "edited", "editor", "world", "earth", "open", "prix", "turismo", "asian", "martial", "arts", "painting", "graphics", "3d", "landscape", "rude", "gore", "guro", "chan", "4chan", "2chan", "7chan", "futaba", "prank", "funny", "joke", "soda", "ad", "advertisement", "ritual", "origami", "common", "pre-marital", "modeling", "play", "dress", "clothing", "wear", "formal", "outfit", "buddah", "christ", "savior",
        "band", "musician", "otaku", "slutty", "stripper", "pole", "dancing", "guitar", "hero", "freaks", "j-pop", "j-rock", "ayumi", "hamasaki", "utada", "hikaru", "movement", "revolution", "revolt", "spanish", "inquisition", "fallacy", "community", "communist", "revolver", "gun", "shotgun", "hunting", "deer", "bear", "dick", "cheney", "condoleeza", "late", "nite", "early", "morning", "hell", "retribution", "heaven", "free", "hacks", "cracks", "hax", "hax0r", "glitched", "glitches", "tricks", "cheats", "cheater", "cheated", "unfair", "code", "codes", "programming", "language", "languages", "tongue", "eye", "heart", "ear", "foot", "toe", "medical", "medicene", "doctor", "hand", "finger", "leg", "arm", "shoulder", "elbow", "wrist", "knee", "cap", "bone", "pelvis", "skull", "brain", "nose", "cartilage", "mortality", "mortal", "coil", "spring", "object", "everyday", "lord", "king", "savior", "emperor", "story", "segue", "van", "automobile", "automatic", "country", "hick", "redneck", "hillbillie", "hillbilly", "stockton", "sacramento", "dallas", "san", "francisco", "diego", "jose", "puto", "taco", "bell", "border", "patrol", "mcdonalds", "KFC", "kentucky", "fried", "chicken", "trans", "fat", "saturated", "monounsaturated", "polyunsaturated", "unsaturated", "peanuts", "food", "network", "oils", "cooking", "vine", "garden", "peaches", "fruits", "apples", "pears", "grapes", "banana", "infamous", "murderer", "killer", "victim", "historian", "history", "historical", "account", "yes", "no", "maybe", "so", "i", "dont", "know", "homo", "homosexual", "immigrant", "irrigated", "irrigation", "pipe", "sewer", "mutation", "ninja", "chemical", "waste", "plant", "grow", "shrink", "sun", "light", "darkness", "scene", "the", "cd", "chip", "micro", "mini", "macro", "quantum", "warp", "space", "drive", "disk", "star", "trek", "pizza", "hut", "dominos", "ecoli", "e.coli", "ebola", "coli", "anthrax", "threat", "nani", "watashi", "anata", "kimi", "boku", "baka", "hikari", "gizmo", "lingo", "GSN", "scrapin", "rapper", "chains", "mine", "yours", "my", "hers", "his", "everyones", "everybody", "starlet", "diva", "awards", "faggotz", "red", "yellow", "orange", "black", "blue", "white", "green", "purple", "violet", "ultra", "spectrum", "speculum", "plug", "o"])

MIN_SPAM_WORDS = 80
CANDIDATE_SEPARATORS =  [ 'and', '\'', '"', ':', '.', ';', ',']
PHRASE, LANG, CULTURE, PRIMARY, WEIGHT = list(range(5))
COMMA = ','

BAD_AK_PHRASES = [' episode ', ' ep ', 'movie', 'series']

def vtv_non_alphanumeric_to_space(s):
    return s.translate(vtv_non_alphanumeric_to_space_table)

def vtv_non_alphanumeric_to_space_and_lower(s):
    return s.translate(vtv_non_alphanumeric_to_space_and_lower_table)

def vtv_remove_non_printable(s):
    return s.translate(vtv_non_printable_table)

def vtv_remove_non_alphanumeric(s):
    return s.translate(vtv_non_alphanumeric_table)

def vtv_remove_space(s):
    return s.translate(string.whitespace)

def vtv_compress_space(s):
    s = s.strip()
    return compressSpaceRe.sub(' ', s)

def removeSpace(text):
    return vtv_remove_space(text)

def removeExtraSpace(text):
    text = text.strip()
    return re.sub('[ \t\n]+', ' ', text)
    # do we need to replace with space or any on the whitespace

def capitalize(text):
    return string.capwords(text)

def isSpam(de_string):
    """
        Return True, if given string is spam.
    """
    #If Desc doesnt have any periods (fullstops) and Desc has more than 20 words then consider only title [ignore desc]
    word_count = de_string.count(' ') + 1
    if de_string.find('.') == -1 and word_count > 50:
        return True
    if not de_string.isupper():
        #If  total # of capitalized words in Desc > 20 (or some appropriate), then consider only title (Not applicable to all caps situations)
        capitalized_word_count = len(capitalize_pat.findall(de_string))
        if capitalized_word_count > 85:
            return True
    lower_str = de_string.lower()
    all_words =  set(lower_str.split())
    spam_w_count = all_words.intersection(spam_words)
    if len(spam_w_count) > MIN_SPAM_WORDS:
        return True

    return False


numeric_replacement_map = (('1', 'I', 'One'), ('2', 'II', 'Two'), ('3', 'III', 'Three'), ('4', 'IV', 'Four'), ('5', 'V', 'Five'), ('6', 'VI', 'Six'), ('7', 'VII', 'Seven'), ('8', 'VIII', 'Eight'), ('9', 'IX', 'Nine'), ('10', 'X', 'Ten'), ('11', 'XI', 'Eleven'), ('12', 'XII', 'Twelve'), ('13', 'XIII', 'Thirteen'), ('14', 'XIV', 'Fourteen'), ('15', 'XV', 'Fifteen'), ('16', 'XVI', 'Sixteen'), ('17', 'XVII', 'Seventeen'), ('18', 'XVIII', 'Eighteen'), ('19', 'XIX', 'Nineteen'), ('20', 'XX', 'Twenty'))
numeric_replacement_hash = {}
numeric_pattern = None

def populateNumericReplacementHash():
    global numeric_replacement_hash, numeric_replacement_map, numeric_pattern
    for digit, roman, alphabet in numeric_replacement_map:
        val = alphabet.lower()
        data_tuple = (digit, val, roman.lower())
        numeric_replacement_hash[digit] = data_tuple
        numeric_replacement_hash[roman.lower()] = data_tuple
        numeric_replacement_hash[val] = data_tuple
    numeric_pattern = re.compile(r'\b(%s)\b' % ('|'.join(list(numeric_replacement_hash.keys()))))

def getNumericVariants(kwd):
    global numeric_replacement_hash
    if not numeric_replacement_hash:
        populateNumericReplacementHash()
    kwd_list = ['']
    last_index = 0
    for mObj in numeric_pattern.finditer(kwd):
        sindex = mObj.start()
        eindex = mObj.end()
        digit, val, roman = numeric_replacement_hash.get(kwd[sindex:eindex], ('', '', ''))
        if not digit:
            continue
        new_kwd_list = []
        for text in kwd_list:
            str_digit = "%s%s%s"%(text, kwd[last_index: sindex], digit)
            str_val = "%s%s%s"%(text, kwd[last_index: sindex], val)
            str_roman = "%s%s%s"%(text, kwd[last_index: sindex], roman)
            new_kwd_list.extend([str_digit, str_val, str_roman])
        kwd_list=new_kwd_list
        last_index = eindex

    if not last_index:
        return [kwd]
    if last_index == len(kwd):
        return kwd_list

    new_kwd_list = []
    for text in kwd_list:
        complete_str = "%s%s"%(text, kwd[last_index:])
        new_kwd_list.append(complete_str)
    return new_kwd_list

def get_bracketed_string(title):
    match_obj = BRACKET_STRING_RE.match(title)
    bracketed_str = ''
    if match_obj:
        data_dict = match_obj.groupdict()
        cleaned_title = data_dict['text'].strip()
        if cleaned_title:
            title = cleaned_title
            bracketed_str = data_dict['br_text'].strip()
    return (title, bracketed_str)

class cleanVariantsEngine:
    def __init__(self, alternate_spell_file = None):
        self.as_dict = {}
        self.initializeAsDict(alternate_spell_file)
        self.articlePattern = re.compile (r'\b(a|an|the)\b')

    def initializeAsDict(self, alternate_spell_file):
        if not alternate_spell_file:
            return
        self.reg_exp = re.compile(r'Ti: (?P<title>.*?)#<>#As: (?P<spells>.*)$')
        as_file = open(alternate_spell_file)
        for line in as_file:
            line = line.strip()
            for mO in self.reg_exp.finditer(line):
                title = mO.group('title')
                spells = mO.group('spells')
                spells = spells.split('<>')
                title = title.strip().lower()
                for spell in spells:
                    self.as_dict[spell.strip().lower()] = title

    def get_as_base_variant(self, variant):
        variant = vtv_compress_space(variant)
        variant_as = ''
        for word in variant.split(" "):
            word = self.as_dict.get(word, word)
            variant_as += word
            variant_as += ' '
        return variant_as.strip()

    def getStrippedSpaceVariant(self, kwd):
        return re.sub('\s*', '', kwd)

    def getNormalizedCleanVariants(self, kwd):
        variant_list = []
        kwd = kwd.strip().lower()
        kwd = ampercentRe.sub(' and ', kwd)
        kwd = backslashWRe.sub(' with ', kwd)
        kwd = genericMergeLib.convert_latin1_ascii(kwd)
        kwd = str(kwd).strip()
        kwd_ap1 = kwd.replace("'s", " ")
        kwd_ap1 = kwd_ap1.replace("`s", " ")
        kwd_ap2 = kwd.replace("'s", "s")
        kwd_ap2 = kwd_ap2.replace("'s", "s")
        kwd_ap3 = kwd.replace("'", "")
        kwd_ap3 = kwd_ap3.replace("`", "")
        processed_nodes_1 = []
        temp_variant_list = [kwd, kwd_ap1, kwd_ap2, kwd_ap3]
        temp_variant_list = getDotVariants(temp_variant_list)
        for variant in temp_variant_list:
            if variant in processed_nodes_1:
                continue
            processed_nodes_1.append(variant)
            variant_and1 = variant.replace('&', ' and ')
            variant_and2 =  COMMON_AND_RE.sub(' & ', variant)
            variant_as = self.get_as_base_variant(variant)
            variant_article = self.articlePattern.sub(' ', variant)
            variant_article_as = self.articlePattern.sub(' ', variant_as)
            processed_nodes_2 = []
            for cur_var in (variant, variant_as, variant_article, variant_article_as, variant_and1, variant_and2):
                cur_var = vtv_non_alphanumeric_to_space(cur_var)
                cur_var = vtv_compress_space(cur_var)
                if cur_var in processed_nodes_2:
                    continue
                processed_nodes_2.append(cur_var)
                variant_sp = self.getStrippedSpaceVariant(cur_var)
                if variant_sp not in variant_list:
                    variant_list.append(variant_sp)
                if cur_var not in variant_list:
                    variant_list.append(cur_var)
        return variant_list

articlePattern = re.compile (r'\b(a|an|the)\b')
def getCleanVariants(s):
    variant_list = []
    s = s.strip()
    s = genericMergeLib.convert_latin1_ascii(s)
    s = str(s).strip()
    s = s.replace('&amp;', '&')
    s = backslashWRe.sub(' with ', s)
    orig_s = s
    s = s.lower()
    if len(s) > 50:
        s1 = vtv_non_alphanumeric_to_space(s)
        s1 = vtv_compress_space(s1)
        variant_list.append(s1)
        variant_list.append(s1.replace(' ', ''))
        return list(set(variant_list))
    #Variants for epostrophe
    s1 = s.replace("'s ", " ")
    s1 = s1.replace("`s ", " ")
    s2 = s.replace("'", "")
    s2 = s2.replace("`", "")
    if s1 == s:
        s1 = None
    if s2 == s:
        s2 = None
    temp_variant_list = [orig_s, s, s1, s2]
    temp_variant_list = getDotVariants(temp_variant_list)
    for variant in temp_variant_list:
        if not variant:
            continue
        variant_and = variant.replace('&', ' and ')
        variant_symbolic_and =  COMMON_AND_RE.sub(' & ', variant)
        if variant_and == variant:
            variant_and = None
        if variant_symbolic_and == variant:
            variant_symbolic_and = None
        for variant_1 in (variant, variant_and, variant_symbolic_and):
            if not variant_1:
                continue
            variant_1 = vtv_non_alphanumeric_to_space(variant_1)
            variant_article = articlePattern.sub(' ', variant_1)
            if variant_article == variant_1:
                variant_list_temp = (variant_1, )
            else:
                variant_list_temp = (variant_1, variant_article)
            for variant_2 in variant_list_temp:
                if not variant_2:
                    continue
                variant_numeric_list = getNumericVariants(variant_2)
                for variant_3 in variant_numeric_list:
                    variant_3 = vtv_compress_space(variant_3)
                    if variant_3:
                        variant_list.append(variant_3)
                        variant_list.append(variant_3.replace(' ', ''))
    return list(set(variant_list))

def getVariantsWithoutArticleVariants(s):
    variant_list = []
    s = s.strip()
    s = genericMergeLib.convert_latin1_ascii(s)
    s = str(s).strip()
    s = s.replace('&amp;', '&')
    s = backslashWRe.sub(' with ', s)
    orig_s = s
    s = s.lower()
    if len(s) > 50:
        s1 = vtv_non_alphanumeric_to_space(s)
        s1 = vtv_compress_space(s1)
        variant_list.append(s1)
        variant_list.append(s1.replace(' ', ''))
        return list(set(variant_list))
    #Variants for epostrophe
    s1 = s
    if s.endswith("'s"):
        s1 = s.replace("'s", " ")
    elif s.endswith("`s"):
        s1 = s.replace("`s", " ")

    s2 = s.replace("'", "")
    s2 = s2.replace("`", "")
    if s1 == s:
        s1 = None
    if s2 == s:
        s2 = None
    temp_variant_list = [orig_s, s, s1, s2]
    temp_variant_list = getDotVariants(temp_variant_list)
    for variant in temp_variant_list:
        if not variant:
            continue
        variant_and = variant.replace('&', ' and ')
        variant_symbolic_and =  COMMON_AND_RE.sub(' & ', variant)
        if variant_and == variant:
            variant_and = None
        if variant_symbolic_and == variant:
            variant_symbolic_and = None
        for variant_1 in (variant, variant_and, variant_symbolic_and):
            if not variant_1:
                continue
            variant_1 = vtv_non_alphanumeric_to_space(variant_1)
            if variant_1:
                variant_numeric_list = getNumericVariants(variant_1)
                for variant_3 in variant_numeric_list:
                    variant_3 = vtv_compress_space(variant_3)
                    if variant_3:
                        variant_list.append(variant_3)
                        variant_list.append(variant_3.replace(' ', ''))
    return list(set(variant_list))

def getDotVariants(variant_list):
    if not variant_list:
        return
    primary_string = variant_list[0]
    variant1 = replaceStringIfProxmityAreInitials(primary_string, ".").lower()
    for each_variant in variant_list:
        if each_variant == variant1:
            break
    else:
        variant_list.append(variant1)
    variant2 = replaceStringIfProxmityAreInitials(primary_string, ". ").lower()
    for each_variant in variant_list:
        if each_variant == variant2:
            break
    else:
        variant_list.append(variant2)
    return variant_list[1:]

def is_initial(primary_string, index, replace_string):
    if index == 0:
        return False
    if index == len(primary_string) - 1:
        return False
    if primary_string[index-1].isupper() and primary_string[index+len(replace_string)].isupper():
        if (index == 1 or not primary_string[index-2].isalnum()) and (index == len(primary_string) - len(replace_string) - 1  or not primary_string[index+len(replace_string) + 1].isalnum()):
            return True
    return False

def replaceStringIfProxmityAreInitials(primary_string, replace_string):
    index = primary_string.find(replace_string)
    variant = ''
    while(index != -1):
        replace_flag = is_initial(primary_string, index, replace_string)
        variant = "%s%s" %(variant, primary_string[:index])
        if not replace_flag:
            variant = "%s%s" %(variant, replace_string)
        primary_string = primary_string[index+len(replace_string):]
        index = primary_string.find(replace_string)
    variant = "%s%s" %(variant, primary_string)
    return variant

epititle_splitter_format = re.compile(r'[/;:]')
def get_epititle_variants(epititle):
    variants = epititle_splitter_format.split(epititle)
    if len(variants) > 1:
        variants.append(epititle)
    return variants

title_splitter_format = re.compile(r':|\.\.\.')
def get_title_variants(title):
    variants = title_splitter_format.split(title)

    if len(variants) <= 1:
        return []
    return variants

def removeSpecialSymbols(s):
    s = genericMergeLib.convert_latin1_ascii(s)
    s = str(s).strip()
    s = ampercentRe.sub(' and ', s)
    s = backslashWRe.sub(' with ', s)
    s = vtv_non_alphanumeric_to_space(s)
    s = vtv_compress_space(s)

    return s

def oldcleanString(s, make_lower=True):
    s = genericMergeLib.convert_latin1_ascii(s, False)
    if make_lower:
        s = s.lower()
    s = s.replace("'s ", " ")
    s = s.replace("`s ", " ")
    s = s.replace("'", "")
    s = s.replace("`", "")
    s = ampercentRe.sub(' and ', s)
    s = backslashWRe.sub(' with ', s)
    s = str(s).strip()
    s = vtv_non_alphanumeric_to_space(s)
    s = vtv_compress_space(s)
    return s

punctuation_str = '"\'.?!:;-_()[]"/,'
regex = re.compile('[%s]' % re.escape(punctuation_str))
def cleanPunctuationsFromString(s, make_lower=True):
    s = genericMergeLib.convert_latin1_ascii(s, False)
    if make_lower:
        s = s.lower()
    s = regex.sub(' ', s)
    s = str(s).strip()
    s = vtv_compress_space(s)
    return s

def checkLength(tup, length):
    if not tup or len(tup)==length:
        return False
    return True

def serialize_phrase(phrase, attr_list):
    """
    Serialize the attribute list for a phrase list.

    >>>> attr_list = [[u"eng", u"", u"", u"", u""], [u"fra", u"", u"", u"", u""], ]
    >>>> print serialize_phrase(u"Titanic", attr_list)
    u'Titanic{eng,fra}'

    """
    if not all([len(x) == 5 for x in attr_list]):
        raise Exception("attr list length should be 5")
    attr_count = len(attr_list)
    remap_list = [list() for _ in range(5)]
    for each in attr_list:
        for idx, each_elm in enumerate(each):
            remap_list[idx].append(each_elm)

    final = []
    count_non_empty = 0
    for idx, each in enumerate(remap_list):
        if any(each):
            if idx == 3:
                final.append(COMMA.join(['|'.join(attr) for attr in each]))
            else:
                final.append(COMMA.join(each))
            count_non_empty += 1
        else:
            final.append('')
    if count_non_empty == 0:
        raise Exception("Lang is empty, other attr also empty")
    elif count_non_empty == 1:
        if final[0]:
            lang_paired_string = "%s{%s}" % (phrase, final[0])
        else:
            raise Exception("Lang is empty, other attr non empty")
    else:
        lang_paired_string = "%s{%s:%s:%s:%s:%s}" % tuple([phrase] + final)
    return lang_paired_string

def generateLangPairedString(values_list):
    phrase, lang, culture, region, attributes, weight = values_list
    if checkLength(culture, len(lang)) or checkLength(region, len(lang)) or checkLength(attributes, len(lang)) or checkLength(weight, len(lang)):
        raise Exception('The arguments are not in the correct format')
    if lang and not culture and not region and not attributes and not weight:
        lang_paired_string = "%s{%s}" % (phrase, COMMA.join(lang))
    elif lang and not culture and not region and not attributes and set(weight) == set(['100',]):
        lang_paired_string = "%s{%s}" % (phrase, COMMA.join(lang))
    else:
        weight = normalize_weight(weight)
        lang_paired_string = "%s{%s:%s:%s:%s:%s}" % (phrase, COMMA.join(lang), COMMA.join(culture), COMMA.join(region), COMMA.join(['|'.join(attr) for attr in attributes]), COMMA.join(weight))
    return lang_paired_string

def normalize_weight(weight):
    w_list = []
    for w in weight:
        if w == '100':
            w_list.append('')
        else:
            w_list.append(w)

    if (len(w_list) == 1):
        weight = w_list
    elif (len(set(w_list)) == 1):
        weight = []
    else:
        weight = w_list

    return weight

lang_paired_rgx = re.compile(r'(?P<phrase>[^\}\{]+)\{(?P<lang>[^\<\>:]+):?(?P<culture>[^\<\>:]*):?(?P<region>[^\<\>:]*):?(?P<attribute>[^\<\>:]*):?(?P<weight>[^\<\>:]*)\}')
def parseMultiValuedLangPairedString(string, separator='<>', only_title=False, min_weight=0):
    '''
    Returns a generator that yields individual phrase+meta variants
    >>>> s = u"The Amazing World of Gumball{eng,spa,ind,hbs,fil,cat:,,,srp,,:::}<>Gumball csodÃ¡latos vilÃ¡ga{hun"
    >>>> parseMultiValuedLangPairedString(s)
       <generator object parseMultiValuedLangPairedString at 0x10ba66c30>
    >>>> for i in parseMultiValuedLangPairedString(s):
       ...:     print i
          ...:
      [u'The Amazing World of Gumball', u'eng', u'', u'', [], u'']
      [u'The Amazing World of Gumball', u'spa', u'', '', [], '']
      [u'The Amazing World of Gumball', u'ind', u'', '', [], '']
      [u'The Amazing World of Gumball', u'hbs', u'srp', '', [], '']
      [u'The Amazing World of Gumball', u'fil', u'', '', [], '']
      [u'The Amazing World of Gumball', u'cat', u'', '', [], '']
      [u'Gumball csod\xe1latos vil\xe1ga', u'hun', '', '', [], '']
    '''
    if not string:
        return
    for pair in string.split(separator):
        if '{' not in pair:
            phrase, lang_list, culture_list, region_list, attribute_list, weight_list = pair, ['eng'], [], [], [], []
        else:
            phrase, attrib = pair.rsplit('{', 1)
            attrib = attrib.strip('}').split(':')
            if (len(attrib) == 1):
                lang_list, culture_list, region_list, attribute_list, weight_list = attrib[0].split(COMMA), [], [], [], []
            else:
                lang_list, culture_list, region_list, attribute_list, weight_list = attrib[0].split(COMMA), attrib[1].split(COMMA), attrib[2].split(COMMA), attrib[3].split(COMMA), attrib[4].split(COMMA)
        for index, lang in enumerate(lang_list):
            if index >= len(culture_list):
                culture_list.append('')
            if index >= len(region_list):
                region_list.append('')
            if index >= len(attribute_list):
                attribute_list.append([])
            if index >= len(weight_list):
                weight_list.append('')
            if not attribute_list[index]:
                attr = []
            else:
                attr = attribute_list[index].split('|')
            if min_weight and weight_list[index] and int(weight_list[index]) < min_weight:
                continue
            if only_title:
                yield phrase
            else:
                yield [phrase, lang, culture_list[index], region_list[index], attr, weight_list[index]]

def parseLangPairedString(lang_paired_string):
    components = lang_paired_rgx.match(lang_paired_string)
    phrase, lang, culture, region, attribute, weight = components.group('phrase'), components.group('lang'), components.group('culture'), components.group('region'), components.group('attribute'), components.group('weight')
    return (phrase, lang, culture, region, attribute, weight)

def getPhraseForLanguageFromLangPairString(string, _lang, only_title = False, separator = '<>', min_weight=0):
    for phrase, lang, culture, region, attr, weight in parseMultiValuedLangPairedString(string, separator, min_weight=min_weight):
        if lang == _lang:
            if only_title:
                yield phrase
            else:
                yield [phrase, lang, culture, region, attr, weight]

title_score_re = re.compile( r'(?P<title>.*?)({(?P<score>[^{}]*?)})?($|<>)' )
title_score_semicol_re = re.compile( r'(?P<title>.*?)({(?P<score>[^{}]*?)})?($|;)' )
def parseMultiValuedVtvString(s, only_titles=False, separator='<>'):
    parse_re = None
    if separator == '<>':
        parse_re = title_score_re
    elif separator == ';':
        parse_re = title_score_semicol_re
    else:
        raise Exception('Unknown separator for parseMultiValuedVtvString')

    for mo in parse_re.finditer(s):
        tmp_title, tmp_score = mo.group('title'), mo.group('score')
        tmp_title = tmp_title.strip()
        if not tmp_title:
            continue
        if only_titles:
            yield tmp_title
        else:
            yield (tmp_title, tmp_score)

title_score_attr_re = re.compile(r'(?P<title>.*?)({(?P<score>[^{}]*?)})?({(?P<attr>[^{}]*?)})?$')
def parseMultiValuedAttributedVtvString(s, only_titles=False, separator='<>'):
    for each_string in [_f for _f in s.split(separator) if _f]:
        mo = title_score_attr_re.match(each_string)
        title = mo.group('title').strip()
        if only_titles:
            yield title
        else:
            yield mo.group('title'), mo.group('score'), mo.group('attr')

title_score_pair_re = re.compile( r'(?P<title1>.*?)({(?P<score1>[^{}]*?)})?<>(?P<title2>.*?)({(?P<score2>[^{}]*?)})?($|<<>>)' )
def parseMultiValuedPairedVtvString(s, only_titles=False, separator='<<>>'):
    for mo in title_score_pair_re.finditer(s):
        tmp_title1, tmp_score1 = mo.group('title1'), mo.group('score1')
        tmp_title2, tmp_score2 = mo.group('title2'), mo.group('score2')
        tmp_title1 = tmp_title1.strip()
        tmp_title2 = tmp_title2.strip()
        if not (tmp_title1 or tmp_title2):
            continue
        if only_titles:
            yield (tmp_title1, tmp_title2)
        else:
            yield (tmp_title1, tmp_score1, tmp_title2, tmp_score2)

def generateMultiValuedPairedVtvString(list_of_iterables, separator='<<>>'):
    '''
    generateMultiValuedPairedVtvString([("tom cruise", "wiki1", "somerole", "wiki2"),("MrBean", "wiki3", "", "")])
    returns "tom cruise{wiki1}<>somerole{wiki2}<<>>MrBean{wiki3}<>"
    '''
    pair_list = []
    for c, cg, r, rg in list_of_iterables:
        comp1, comp2 = '', ''
        if c and cg:
            comp1 = "%s{%s}" %(c, cg)
        if r and rg:
            comp2 = "%s{%s}" %(r, rg)
        if comp1 or comp2:
            pair = '%s<>%s' % (comp1, comp2)
            pair_list.append(pair)
    return separator.join(pair_list)

def generateMultiValuedVtvString(list_of_iterables, separator='<>', limit=None):
    """
    based on the length of the tuples generates a string which can be parsed by
    parseMultiValuedVtvString and parseMultiValuedAttributedVtvString
    for parseMultiValuedAttributedVtvString it should be 2
    >>> generateMultiValuedVtvString([("tom cruise", "wiki1"), ("kate", ), ("Spielberg", "wiki3", "director")])
    'tom cruise{wiki1}<>kate<>Spielberg{wiki3}{director}'
    Given a list of tuples say
    [ ( "a",  ), ( "a", None, None ), ( "a", "b", None ),
      ( "a", "b", "c" ), ( "a", None, "c"), ( "a", "b", "c", "d", "e")
    ]
    a<>a<>a{b}<>a{b}{c}<>a{}{c}<>a{b}{c}{d}{e}
    if a limit is given - it limits then entries in curly braces
    i.e. from tuple[1: limit + 1]
    for limit = 2 a{b}{c}{d}{e} will be a{b}{c}
    for limi = 1 it will be a{b}
    """

    str_list = []
    if limit is not None and isinstance(limit, int):
        list_of_iterables = [ tup[: limit + 1 ] for tup in list_of_iterables ]

    for tup in list_of_iterables:
        tup = list(tup)
        title = tup[0]
        curly_brace_contents = tup[1: ]
        curly_brace_contents.reverse()

        curly_brace_contents_list = []
        has_data = False
        for i in curly_brace_contents:
            if not i:
                if not has_data:
                    continue
                i = ''
            has_data = True
            curly_brace_contents_list.append("{%s}" % i)
        curly_brace_contents_list.reverse()

        str_list.append('%s%s' % (title, ''.join(curly_brace_contents_list)))

    return separator.join(str_list)

def getCleanName(input_str):
    name = input_str.lower()
    name = genericMergeLib.convert_latin1_ascii(re.sub(',|-|\.| |\'|"', '', name), False)
    return name.strip()

def generateSubstrings(inputString, min_count, max_count, sw_obj=None):
    set_list = []
    tempWordList = inputString.split()
    maxWordCnt = len(tempWordList)
    tempHash = {}

    for n in range(min(max_count, maxWordCnt), min_count, -1):
        list_strings = [" ".join(tempWordList[no:no+n]) for no in range(maxWordCnt-n+1)]

        if n == 1 and sw_obj:
            list_strings = [x for x in list_strings if not sw_obj.isStopWord(x)]

        tempHash[n] = list_strings
        set_list.extend(list_strings)

    return (tempHash, frozenset(set_list))

def isKwOnTi(kw, ti, flexibility):
    # I am expecting kw and ti cleaned
    if flexibility == 10:
        kw=removeSpace(kw)
        pat=''
        for c in kw:
            pat+= c+r'[\s]?'
        repat=re.compile(r'(^|\b)'+pat+r'(\b|$|s)')
        if repat.search(ti):
            return 1
        else:
            return None
    elif flexibility == 5:
        kw=kw.split(' ')
        ti=ti.split(' ')
        for k in kw:
            if k not in ti:
                return None
        return 1
    else:
        return None


def getSplitScoreWithSep( sep, inString):
    maxChunkStr, maxChunkLen, emptyChunks = '___', -1, 0
    chunkList = []
    for chunk in inString.split(sep):
        if len(chunk.strip()) > 0:
            chunkList.append ( chunk.strip())
            if len(chunk.strip()) > maxChunkLen:
                maxChunkLen = len(chunk.strip())
                maxChunkStr = chunk.strip()
        else:
            emptyChunks += 1
    if len(chunkList):
        score = abs ( COMMON_LEN - (len(inString)/(emptyChunks+len(chunkList)) ) )
        print(("Sep: ", sep, " Score: ", score))
    else:
        score = -1
    return score, maxChunkStr, maxChunkLen, chunkList

def furtherCleanChunks( inList=[]):
    retList = []
    for item in inList:
        tmpItem = cleanString(item)
        if tmpItem.startswith('and '):
            retList.append( item[4:].strip())
        elif tmpItem.startswith('aka '):
            retList.append( item[4:].strip())
        else:
            retList.append( item)
    return retList

def betterSeparator( sep1, sep2):
    index1, index2 = -1, -1
    if sep1 in CANDIDATE_SEPARATORS:
        index1 = CANDIDATE_SEPARATORS.index(sep1)
    if sep2 in CANDIDATE_SEPARATORS:
        index2 = CANDIDATE_SEPARATORS.index(sep2)
    if index1 >= index2:
        return sep1
    else:
        return sep2

def smartSplitString( inString='', doRecurse=True, prePrune=True, seperatorInfo=False):
    bestScore, bestSep, bestChunkList = 1000, ['___'], []
    inString = inString.strip()
    inLength = len(inString)
    if not inLength:
        return bestScore, bestSep, bestChunkList
    #
    if prePrune:
        inString = singleQuoteRE.sub( '', inString)
        inString = bracketRE.sub( '', inString)
        inLength = len(inString)
    #
    tmpString = trialRE.sub( ' ', inString)
    sepList = []
    for sep in tmpString.split(' '):
        if len(sep) and sep not in sepList:
            sepList.append( sep)
    extraSepList = [ '\r', '\n']
    if inString.find('and') != -1:
        extraSepList = ['and']
    if not len(sepList + extraSepList):
        return 0, [], [ inString]
    for sep in set(sepList + extraSepList):
        score, chunkList, emptyChunks = 1000, [], 0
        for chunk in inString.split(sep):
            chunk = chunk.strip()
            if doRecurse and len(chunk) > COMMON_LEN + 5:
                tmpScore, tmpSep, tmpChunkList = smartSplitString( chunk, doRecurse=False, prePrune=prePrune, seperatorInfo=seperatorInfo)
                chunkList.extend( tmpChunkList)
            elif len(chunk) > 0:
                if seperatorInfo:
                    chunkList.append( (sep, chunk))
                else:
                    chunkList.append( chunk)
            else:
                emptyChunks += 1
        numChunks = len(chunkList) + emptyChunks
        if numChunks:
            score = abs ( COMMON_LEN - inLength/numChunks)
        if score < bestScore or ( score == bestScore and sep == betterSeparator( bestSep[0], sep)):
            bestScore = score
            bestSep[0] = sep
            bestChunkList = chunkList
    bestChunkList = furtherCleanChunks( bestChunkList)
    return bestScore, bestSep, bestChunkList

def preProcessString(s):
    s = genericMergeLib.convert_latin1_ascii(s, False)
    s = SPECIAL_CHAR_RE.sub(' ', s)
    s = DIGIT_SPECIAL_CHAR_RE.sub(' ', s) #Skipping 1,000 , 22:30
    return s

def IsBadPhrase(phrase, field, vt):
    if any(y in phrase.lower() for y in BAD_AK_PHRASES):
        if (vt in (CONTENT_TYPE_TVSERIES, CONTENT_TYPE_MOVIE, CONTENT_TYPE_TVVIDEO, CONTENT_TYPE_EPISODE)
                and field in ('Ti', 'Ak')):
            return True
    return False

class TestCleanString(unittest.TestCase):
    """
    Test cases to test cleanString function.
    """
    def setUp(self):
        pass

    def test_163266(self):
        """
        Checkin: 163266
        Ticket: 25252
        Description: cleanString should substitute space for unknown characters.
        """
        orig_string = "Veveo公司India"
        expected = "veveo india"
        self.assertEqual(cleanString(orig_string), expected)


def main(argv):
    """
    print isKwOnTi(argv[1],argv[2], 5)

    print "ASCII: ", vtv_ascii_table
    print "Non Printable: ", vtv_non_printable_table
    print "Non Alpha Numeric: ", vtv_non_alphanumeric_table
    print "Non Space: ", vtv_non_space_table
    """

    print("Alpha numeric to space: ", vtv_non_alphanumeric_to_space_table)
    print("Alpha numeric to space_lower table: ", vtv_non_alphanumeric_to_space_and_lower_table)

    str_list = [ "aBcDeFGii1-2", "aBcDeFGii1 2", "aBcDeFGii1@3", "aBcDeFGii1\0052", 'Yasuhiro Nakasone']

    print("Printable Characters")
    for s in str_list:
        print(s, " ", vtv_remove_non_printable(s))

    print("AlphaNumeric Characters")
    for s in str_list:
        print(s, " ", vtv_remove_non_alphanumeric(s))

    print("AlphaNumeric To Space Characters")
    for s in str_list:
        print(s, " ", vtv_non_alphanumeric_to_space(s))

    print("AlphaNumeric To Space and Lower Characters")
    for s in str_list:
        print(s, " ", vtv_non_alphanumeric_to_space_and_lower(s))

    print("Non Space Characters")
    for s in str_list:
        print(s, " ", vtv_remove_space(s))

    print("Generate sub strings")
    print(generateSubstrings("abc def fgh ghi jkl", 0, 6))

    str_list.extend(['Lucyna Langer-Ka\\u0142ek', 'Gustavo Mart\xednez Zuvir\xeda', 'junk unicode'])
    print("cleanString")
    for s in str_list:
        print(s, " ", cleanString(s))

    print("parseMultiValuedVtvString")
    vtv_string_list = ["", "title1", "title2{gid2}", "title3<>title4{gid4}", "<>title5{gid5}<>title6{gid6}", "title7{gid7}<>title8{gid8}<>", "title9{gid9}<>title10{gid10}", "title{11{gid11}<>{{{title12}}}{gid12}<>title13{gid13}", "<>"]
    for s in vtv_string_list:
        for title, gid in parseMultiValuedVtvString(s):
            print(title, gid)

def eng_string_transform(s):
    '''
    English Language specific transformations
    '''
    s = stripApostsRe.sub('\\1 ', s)
    s = stripApostOnlyRe.sub('', s)
    s = ampercentRe.sub(' and ', s)
    s = backslashWRe.sub(' with ', s)
    return s


def cleanString(s, make_lower=True, lang='eng',
                allow_punct=False, translate=None):
    """
    lang parameter order of preference
    1. non empty lang parameter
    2. Default English
    Invalid lang value if passed will throw error
    """

    # Force it to empty string for any Falsy type, like None
    if not s:
        s = ''

    if translate:
        s = s.translate(translate)

    char_set_obj = LANG_MAP.get(lang, None)
    if char_set_obj is None:
        raise Exception("%s - Language cleanString not implemented" % lang)

    string_transform_func = None
    if lang == 'eng':
        string_transform_func = eng_string_transform

    s = char_set_obj.lang_clean(s, make_lower,
                                allow_punct, string_transform_func)
    s = s.strip()
    s = vtv_compress_space(s)
    return s


def normalise(s, lang='eng', translate=None):
    '''
    Similar to clean string, but does not discard alpha numeric chars
    of any lang.
    '''
    def normalize_eng_string_transform(s):
        '''
        English Language specific transform,
        '''
        s = stripApostOnlyRe.sub('', s)
        s = ampercentRe.sub(' and ', s)
        s = backslashWRe.sub(' with ', s)
        return s

    if not s:
        s = ''

    if translate:
        s = s.translate(translate)

    char_set_obj = LANG_MAP.get(lang, None)
    if char_set_obj is None:
        raise Exception("%s - Language cleanString not implemented" % lang)

    string_transform_func = None
    if lang == 'eng':
        string_transform_func = normalize_eng_string_transform

    s = char_set_obj.lang_clean(s, True, False,
                                string_transform_func, only_norm=True)
    s = s.strip()
    s = vtv_compress_space(s)
    return s


def is_in_valid_charset(in_string, lang="eng", consider_latin=True, accept_fraction=0.5):
    if not in_string:
        return False
    if lang not in LANG_MAP:
        return True
    if isinstance(in_string, str):
        in_string = in_string.decode('utf8')
    uniq_original_len = len(set(in_string))
    lang_obj = LANG_MAP.get(lang, (None, None, None, None))
    if not consider_latin:
        eng_lang = LANG_MAP.get("eng", (None, None, None, None))
        uniq_charset_len = len(set([i for i in in_string if lang_obj.is_in(i) and not eng_lang.is_in(i)]))
    else:
        uniq_charset_len = len(set([i for i in in_string if lang_obj.is_in(i)]))
    fraction_in_charset = float(uniq_charset_len)/uniq_original_len
    return fraction_in_charset >= accept_fraction

def get_translate_table(not_needed_chars=' `_$& !+-*/^%#@~;"\'\\(),?=[]', replace_char='.'):
    replace_string = replace_char * len(not_needed_chars)
    translate_table = str.maketrans(not_needed_chars, replace_string)
    return translate_table

def remove_spl_chars(raw_string, translate_table=get_translate_table()):
    return raw_string.translate(translate_table).strip()

# Add Americanised version also
normalize = normalise
if __name__ == '__main__':
    main(sys.argv)
    unittest.main()
