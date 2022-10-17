
import random
from faker import Faker


SPOILER_TEMPLATES = [
    "{TEAM_NAME_1} {INT1}-{INT2} {TEAM_NAME_2} {LEAGUE_NAME} {SEASON}",
    "Highlights {TEAM_NAME_1} {INT1}-{INT2} {TEAM_NAME_2} {LEAGUE_NAME} {SEASON}",
    "{TEAM_NAME_1} {INT1}-{INT2} {TEAM_NAME_2} Highlights {LEAGUE_NAME} {SEASON}",
    "{TEAM_NAME_1} ({INT1}-{INT2}) {TEAM_NAME_2} {LEAGUE_NAME} {SEASON}",
    "{TEAM_NAME_1} ({INT1}-{INT2}) {TEAM_NAME_2} {LEAGUE_NAME}",
    "{LEAGUE_NAME} {TEAM_NAME_1} ({INT1}-{INT2}) {TEAM_NAME_2}",
    "Resumen de {TEAM_NAME_1} vs {TEAM_NAME_2} ({INT1}-{INT2}) {LEAGUE_NAME} {SEASON}",
    "Resumen de {TEAM_NAME_1} vs {TEAM_NAME_2} {INT1}-{INT2} {LEAGUE_NAME} {SEASON}",
    "{TEAM_NAME_1} were VERY UNDERWHELMING vs. {TEAM_NAME_2} in the {LEAGUE_NAME} - Julien  | ESPN FC", 
    "{TEAM_NAME_1} were VERY UNDERWHELMING vs. {TEAM_NAME_2} in the {LEAGUE_NAME} - ESPN FC", 
    "{TEAM_NAME_1}-{TEAM_NAME_2} {INT1}-{INT2} | {PLAYER_2} and {PLAYER_1} both score again: Goals & Highlights | Serie A 2022/23",
    "HIGHLIGHTS | {TEAM_NAME_1} vs {TEAM_NAME_2} ({INT1}-{INT2}) | {PLAYER_1} ({INT1}) {PLAYER_2}",
    "{TEAM_NAME_1} {INT1}-{INT2} {TEAM_NAME_2} | {LEAGUE_NAME} Highlights",
    "HIGHLIGHTS: {TEAM_NAME_1} {INT1}-{INT2} {TEAM_NAME_2} | Record-breaking {INT1} goals at {PLACE_1}!",
    "HIGHLIGHTS | {TEAM_NAME_1} v {TEAM_NAME_2} ({INT1}-{INT2}) | {PLAYER_1} scores {PLAYER_2} impresses!",
    "Extended Highlights | {TEAM_NAME_1} {INT1}-{INT2} {TEAM_NAME_2}| {PLAYER_1} and {PLAYER_2} hat-tricks!",
    "Extended Highlights | {TEAM_NAME_1} {INT1}-{INT2} {TEAM_NAME_2}| {PLAYER_1} and {PLAYER_2} hattricks!",
    "Extended Highlights | {TEAM_NAME_1} {INT1}-{INT2} {TEAM_NAME_2}| {PLAYER_1} and {PLAYER_2} hatricks!",
    "{PLAYER_1} & {PLAYER_2} goals not enough at the {PLACE_1}",
    "{PLAYER_1} and {PLAYER_2} make it {INTTEXT1} wins from {INTTEXT1}!",
    "{PLAYER_1} scores HAT-TRICK in {INT10_30} minutes!",
    "{PLAYER_1} scores HATTRICK in {INT10_30} minutes!",
    "{PLAYER_1} Scores First {TEAM_NAME_1} Goals",
    "{TEAM_NAME_1} Stun {TEAM_NAME_2} | {LEAGUE_NAME} Highlights",
    "{TEAM_NAME_1} Stun {TEAM_NAME_2} | {TEAM_NAME_1} {INT1} {TEAM_NAME_2} {INT2} | {LEAGUE_NAME} Highlights",
    "{TEAM_NAME_1} {INT1} x {INT2} {TEAM_NAME_2} ● {YEAR} {LEAGUE_NAME} Extended Goals & Highlights HD",
    "{PLAYER_1} and {PLAYER_2} Braces in Dominant Win ",
    "{PLAYER_1} seals first win in {INTTEXT1} for {TEAM_NAME_1}: Goals & Highlights |",
    "{TEAM_NAME_1} beat {TEAM_NAME_2} by {INT1} runs",
    "Highlights {TEAM_NAME_1} beat {TEAM_NAME_2} by {INT1} runs",
    "{TEAM_NAME_1} beat {TEAM_NAME_2} by {INT1} wickets",
    "Highlights {TEAM_NAME_1} beat {TEAM_NAME_2} by {INT1} wickets",
    "{TEAM_NAME_1} win by {INT_RUNS} runs beating {TEAM_NAME_2}",
    "Highlights {TEAM_NAME_1} win by {INT1} runs beating {TEAM_NAME_2}",
    "{TEAM_NAME_1} win by {INT1} wickets beating {TEAM_NAME_2}",
    "Highlights {TEAM_NAME_1} win by {INT1} wickets beating {TEAM_NAME_2}",
    "{TEAM_NAME_1} celebrates its win over {TEAM_NAME_2}",
    "{PLAYER_1} smashes {INT1} in {INT2} | Highlights {TEAM_NAME_1} vs {TEAM_NAME_2}",
    "{PLAYER_1} smashes ton against {TEAM_NAME_1}",
    "{TEAM_NAME_1} vs {TEAM_NAME_2} super over match Highlights",
    "{INT1_6} sixes in an over | highlights {TEAM_NAME_1} vs {TEAM_NAME_2}",
    "Highlights : {TEAM_NAME_1} vs {TEAM_NAME_2} {INT1}{ST} {FORMAT} | {TEAM_NAME_1} won by {INT2} wickets",
    "Highlights {TEAM_NAME_1} vs {TEAM_NAME_2} | {PLAYER_1} played stormy innings",
    "{PLAYER_1} Incredible {INT1} off {INT2} balls at {PLACE_1} | {TEAM_NAME_1} vs {TEAM_NAME_2}",
    "MOM: {PLAYER_1} - {TEAM_NAME_1} vs {TEAM_NAME_2} | Highlights",
    "{TEAM_NAME_1} grabs {LEAGUE_NAME} title",
    "{PLAYER_1} smashes {INT10_30}-ball fifty | Highlights | {TEAM_NAME_1} vs {TEAM_NAME_2}",
    "{TEAM_NAME_1} crushes {TEAM_NAME_2} match Highlights",
]

NON_SPOILER_TEMPLATES = [
    "Highlights Week {WEEK} - {LEAGUE_NAME} / {SEASON}",
    "Highlights | {TEAM_NAME_1} vs {TEAM_NAME_2}", 
    "Highlights | {TEAM_NAME_1} v {TEAM_NAME_2}", 
    "HIGHLIGHTS | {TEAM_NAME_1} vs {TEAM_NAME_2} {LEAGUE_NAME}",  
    "HIGHLIGHTS | {TEAM_NAME_1} v {TEAM_NAME_2} {LEAGUE_NAME}",  
    "HIGHLIGHTS | {TEAM_NAME_1} vs {TEAM_NAME_2} {LEAGUE_NAME} / {SEASON}",  
    "HIGHLIGHTS | {TEAM_NAME_1} v {TEAM_NAME_2} {LEAGUE_NAME} / {SEASON}",  
    "Extended HIGHLIGHTS | {TEAM_NAME_1} vs {TEAM_NAME_2} {LEAGUE_NAME} / {SEASON}",  
    "Extended HIGHLIGHTS | {TEAM_NAME_1} v {TEAM_NAME_2} {LEAGUE_NAME} / {SEASON}",  
    "Extended HIGHLIGHTS | {TEAM_NAME_1} v {TEAM_NAME_2} {LEAGUE_NAME}",  
    "Extended HIGHLIGHTS | {TEAM_NAME_1} v {TEAM_NAME_2}",  
    "#{INT1} {TEAM_NAME_1} vs #{INT2} {TEAM_NAME_2} | {YEAR} {LEAGUE_NAME} Highlights",
    "All goals Matchday {WEEK} {LEAGUE_NAME} {YEAR}",
    "{INT1}nd T20I | Highlights | {TEAM_NAME_1} Tour Of {TEAM_NAME_2} | {DATE}",
    "{PLAYER_1} Best Goals & Skills {TEAM_NAME_1} {LEAGUE_NAME} {SEASON}",
    "{PLAYER_1} Debut for {TEAM_NAME_1} Highlights {DATE}",
    "{TEAM_NAME_1} vs {TEAM_NAME_2} {INT1}{ST} {FORMAT}",
    "Highlights {TEAM_NAME_1} vs {TEAM_NAME_2} {INT1}{ST} {FORMAT}",
    "{TEAM_NAME_1} vs {TEAM_NAME_2} | match {INT1} highlights | {LEAGUE_NAME} {YEAR} highlights",
    "{LEAGUE_NAME} {YEAR} Match {INT1} {TEAM_NAME_1} vs {TEAM_NAME_2} Highlights - {LEAGUE_NAME} Gaming Series",
    "{TEAM_NAME_1} vs {TEAM_NAME_2} - Match Highlights",
    "{TEAM_NAME_1} VS {TEAM_NAME_2} | Match No {INT1} | {LEAGUE_NAME} {YEAR} Match Highlights | Hotstar Cricket",
    "Extended Highlights {TEAM_NAME_1} vs {TEAM_NAME_2}",
    "Top {INT1} moments of {PLAYER_1}",
    "Top {INT1} sixes in cricket history",
    "{INT1}{ST} {FORMAT} | Highlights | {TEAM_NAME_1} Tour of {TEAM_NAME_2} ",
    "Highlights: {TEAM_NAME_1} v {TEAM_NAME_2}, {PLACE_1} | {FORMAT} {YEAR}",
]

DATE_FORMATS = [
    "%A %d. %B %Y",
    "%d/%m/%y",
    "%d %B %Y",
    "%d. %B %Y"
]

INTTEXTS = [
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE"
    "SIX"
]

ST_ND = [
    "st", "nd", "rd", "th"
]

CRICKET_FORMATS = [
    "ODI","TEST", "T20I", "T20",
]

def get_all_teamnames():
    team_names = set()
    with open('teamnames.txt', 'r') as f:
        for line in f:
            team_names.add(line.strip())
    return team_names


def get_all_league_names():
    league_names = set()
    with open('league_names.txt', 'r') as f:
        for line in f:
            league_names.add(line.strip())
    return league_names


def get_all_player_names():
    player_names = set()
    with open('player_names.txt', 'r') as f:
        for line in f:
            player_names.add(line.strip())
    return player_names


def get_random_year():
    return random.randint(1990, 2023, )



def get_random_season():
    x = random.randint(1992,2023)
    y = str(x+1)
    r = random.randint(0, 1)
    if r == 0:
        y = y[2:]
    
    special_char = chr(random.randint(ord('!'), ord('/')))
    return str(x)+special_char+y


def get_random(s):
    return random.sample(s, 1)[0]


def get_random_week():
    return random.randint(1, 50)




if __name__ == '__main__':
    team_names = get_all_teamnames()
    league_names = get_all_league_names()
    myfaker = Faker()

    data_file_handle = open("data.csv", 'w')
    for templates,c,yn in [(SPOILER_TEMPLATES, 1000, 'y'), (NON_SPOILER_TEMPLATES, 2000, 'n')]:
    # for templates,c,yn in [(SPOILER_TEMPLATES, 10, 'y'), (NON_SPOILER_TEMPLATES, 20, 'n')]:
        for t in templates:
            for i in range(c):
                mock_data = {
                    "TEAM_NAME_1": get_random(team_names),
                    "TEAM_NAME_2": get_random(team_names),
                    "PLAYER_1": myfaker.name(),
                    "PLAYER_2": myfaker.name(),
                    "LEAGUE_NAME": get_random(league_names),
                    "YEAR": get_random_year(),
                    "SEASON": get_random_season(),
                    "WEEK": get_random_week(),
                    "INT1": random.randint(0,11),
                    "INT2": random.randint(0,11),
                    "INT10_30": random.randint(10,30),
                    "INT1_6": random.randint(1,6),
                    "DATE": myfaker.date_between(start_date='-5y', end_date='today').strftime(get_random(DATE_FORMATS)),
                    "PLACE_1": myfaker.location_on_land()[2],
                    "INTTEXT1": get_random(INTTEXTS),
                    "FORMAT": get_random(CRICKET_FORMATS),
                    "ST": get_random(ST_ND),
                    "INT_RUNS": random.randint(1, 300),
                }
                # print(t)
                formatted = t.format(**mock_data)
                formatted = formatted.replace(',', '-')
                data_file_handle.write(f"{formatted},{yn}\n")

    data_file_handle.close()
