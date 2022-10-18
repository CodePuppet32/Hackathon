
import nltk
import re
from .util.StringUtil import normalise


ALLOWED = [
    "NNP", "NNPS"
]
IGNORED = [
   "highlights", "vs", "versus", "v", "year", "league"
]


def get_spoiler_free_text(text):
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    new_s_list = []


    for s, t in tagged:
        n_s = normalise(s)
        if all(x.isdigit() for x in n_s.split(" ")):
            new_s = re.sub('\d', 'x', s)
            new_s_list.append(new_s)
            continue

        if t in ALLOWED or s.lower() in IGNORED:
            new_s_list.append(s)
            continue



    return ' '.join(new_s_list)


if __name__ == "__main__":
    S_LIST = [
        "Shrewsbury Town 4-5 Colchester United KNVB Beker 2011/2012",
        "Scotland men 11-8 Walsall ISC B 1992)1993",
        "Highlights Newcastle United 2-4 Sunderland ISC B 1995'1996",
        "Highlights Middlesbrough 0-6 Stranraer J-League Cup 2004/2005",
        "Rochdale (7-1) Milton Keynes Dons Serie A",
        "Newcastle United beat Arsenal by 6 runs",
        "Chesterfield beat Wycombe Wanderers by 9 runs",
        "Highlights Grimsby Town win by 5 wickets beating Bristol City",
        "Hoffenheim celebrates its win over Livingston",
        "watch highlight India's win against Australia league asia cup",
    ]
    for s in S_LIST:
        t = get_spoiler_free_text(s)
        print(f'{s} ====== {t}')




