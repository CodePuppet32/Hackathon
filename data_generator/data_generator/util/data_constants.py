#!/usr/bin/env python3
# coding: utf-8

###############################################################################
# $Id: data_constants.py,v 1.86 2017/12/07 12:17:09 vinay Exp $
# Copyright(c) 2005 Veveo.tv
###############################################################################

WIKI_GID_PREFIX               = ['WIKI', 'WQQQ', 'WGER', 'WFRA', 'WSPA', 'WSWE', 'WNOR', 'WITA', 'WPOR', 'WDUT', 'WDAN', 'WFIN', 'WTUR', 'WRUS', 'WPOL', 'WJPN', 'WKOR', 'WZHO', 'WCAT', 'WUKR', 'WVIE', 'WIND', 'WHUN', 'WRON', 'WCES', 'WHRV', 'WSRP']
NOT_SUPPORTED_WIKI_GID_PREFIX = ['WARA', 'WBUL', 'WEST', 'WFAS', 'WGRE', 'WHEB', 'WHIN', 'WMSA', 'WSCC', 'WSLK', 'WSLV', 'WTGL', 'WTHA']
OTHER_KG_SOURCE_GID_PREFIX    = ['FRB', 'RV', 'TRV', 'TMDB']
GENRE_PREFIX                  = ['G', 'CG', 'SG', 'DETAIL', 'RVG']
MANUAL_GID_PREFIX             = [ 'ROLE', 'RVROLE', 'EPISODE', 'PERSON', 'KGMVP0446', 'KGMVP0445', 'TOPIC' ]
SPORTS_GID_PREFIX             = ['SPORT', 'STAD', 'TEAM', 'TOU', 'GR', 'PL']
MISC_GID_PREFIX               = ['CGP', 'CHAF', 'LANG', 'RVLANG', 'AWARD', 'AWCE', 'OAC', 'FS', 'PC', 'REGION', 'RVREG']
MUSIC_GID_PREFIX              = ['AR', 'LFMAL', 'LFMAR', 'LFMSO', 'RSO', 'MN', 'MW']
MANUAL_GID_PREFIX_SOURCE      = ['MO', 'TV', 'EP', 'PE', 'RO', 'GE', 'LA', 'RE', 'SP', 'PR']
#seed should have different gid prefix than source(matching pre-requisite)
MANUAL_GID_PREFIX_SEED_MVP    = [ 'MVP%s' % prefix for prefix in MANUAL_GID_PREFIX_SOURCE ]
MANUAL_GID_PREFIX_SEED        = [ 'ACE%s' % prefix for prefix in MANUAL_GID_PREFIX_SOURCE ]

VALID_GID_PREFIX_ORDER        = WIKI_GID_PREFIX + \
                                NOT_SUPPORTED_WIKI_GID_PREFIX + \
                                OTHER_KG_SOURCE_GID_PREFIX + \
                                GENRE_PREFIX + \
                                MANUAL_GID_PREFIX + \
                                SPORTS_GID_PREFIX + \
                                MISC_GID_PREFIX + \
                                MUSIC_GID_PREFIX + \
                                MANUAL_GID_PREFIX_SEED_MVP + \
                                MANUAL_GID_PREFIX_SEED + \
                                MANUAL_GID_PREFIX_SOURCE

#multilang wiki record valid prefix
VALID_WIKI_PARENT_GID_PREFIX = tuple(WIKI_GID_PREFIX)

# Content Type Definitions

CONTENT_TYPE_MOVIE                   = 'movie'
CONTENT_TYPE_MOVIE_NUM               = 1
CONTENT_TYPE_TVSERIES                = 'tvseries'
CONTENT_TYPE_TVSERIES_NUM            = 2
CONTENT_TYPE_TVVIDEO                 = 'tvvideo'
CONTENT_TYPE_TVVIDEO_NUM             = 3
CONTENT_TYPE_EPISODE                 = 'episode'
CONTENT_TYPE_EPISODE_NUM             = 4
CONTENT_TYPE_SEQUEL                  = 'sequel'
CONTENT_TYPE_SEQUEL_NUM              = 5
CONTENT_TYPE_PERSON                  = 'person'
CONTENT_TYPE_PERSON_NUM              = 6
CONTENT_TYPE_ROLE                    = 'role'
CONTENT_TYPE_ROLE_NUM                = 7
CONTENT_TYPE_CHANNEL                 = 'channel'
CONTENT_TYPE_CHANNEL_NUM             = 8
CONTENT_TYPE_CHANNEL_AFFILIATION     = 'channelaffiliation'
CONTENT_TYPE_CHANNEL_AFFILIATION_NUM = 9
CONTENT_TYPE_VOD                     = 'vod'
CONTENT_TYPE_VOD_NUM                 = 10
CONTENT_TYPE_THEATRE                 = 'theatre'
CONTENT_TYPE_THEATRE_NUM             = 11
CONTENT_TYPE_ALBUM                   = 'album'
CONTENT_TYPE_ALBUM_NUM               = 12
CONTENT_TYPE_SONG                    = 'song'
CONTENT_TYPE_SONG_NUM                = 13
CONTENT_TYPE_TRACK                   = 'track'
CONTENT_TYPE_TRACK_NUM               = 14
CONTENT_TYPE_TEAM                    = 'team'
CONTENT_TYPE_TEAM_NUM                = 15
CONTENT_TYPE_TOURNAMENT              = 'tournament'
CONTENT_TYPE_TOURNAMENT_NUM          = 16
CONTENT_TYPE_SPORTS_GROUP            = 'sportsgroup'
CONTENT_TYPE_SPORTS_GROUP_NUM        = 17
CONTENT_TYPE_STADIUM                 = 'stadium'
CONTENT_TYPE_STADIUM_NUM             = 18
CONTENT_TYPE_GAME                    = 'game'
CONTENT_TYPE_GAME_NUM                = 19
CONTENT_TYPE_PHRASE                  = 'phrase'
CONTENT_TYPE_PHRASE_NUM              = 20
CONTENT_TYPE_GENRE                   = 'genre'
CONTENT_TYPE_GENRE_NUM               = 21
CONTENT_TYPE_DECADE                  = 'decade'
CONTENT_TYPE_DECADE_NUM              = 22
CONTENT_TYPE_FILTER                  = 'filter'
CONTENT_TYPE_FILTER_NUM              = 23
CONTENT_TYPE_AWARD                   = 'award'
CONTENT_TYPE_AWARD_NUM               = 24
CONTENT_TYPE_LANGUAGE                = 'language'
CONTENT_TYPE_LANGUAGE_NUM            = 25
CONTENT_TYPE_RATING                  = 'rating'
CONTENT_TYPE_RATING_NUM              = 26
CONTENT_TYPE_SIMILAR                 = 'similar'
CONTENT_TYPE_SIMILAR_NUM             = 27
CONTENT_TYPE_ATTRIBUTE               = 'attribute'
CONTENT_TYPE_ATTRIBUTE_NUM           = 28
CONTENT_TYPE_LEXICAL                 = 'lexical'
CONTENT_TYPE_LEXICAL_NUM             = 29
CONTENT_TYPE_INTERSECTION            = 'intersection'
CONTENT_TYPE_INTERSECTION_NUM        = 30
CONTENT_TYPE_LIST                    = 'list'   # Used in Cablevision - kitchensink
CONTENT_TYPE_LIST_NUM                = 31
CONTENT_TYPE_IPC                     = 'ipc'
CONTENT_TYPE_IPC_NUM                 = 39
CONTENT_TYPE_USER                    = 'user'
CONTENT_TYPE_USER_NUM                = 40
CONTENT_TYPE_PLAYLIST                = 'playlist'
CONTENT_TYPE_PLAYLIST_NUM            = 41
CONTENT_TYPE_SPORT                   = 'sport'
CONTENT_TYPE_SPORT_NUM               = 42
CONTENT_TYPE_REGION                  = 'region'
CONTENT_TYPE_REGION_NUM              = 43
CONTENT_TYPE_PC                      = 'productionhouse'
CONTENT_TYPE_PC_NUM                  = 44
CONTENT_TYPE_OAC                     = 'originalchannel'
CONTENT_TYPE_OAC_NUM                 = 45
CONTENT_TYPE_COMPOSITION             = 'composition'
CONTENT_TYPE_COMPOSITION_NUM         = 46
CONTENT_TYPE_TVSERIES_SEQUEL         = 'tvseries_sequel'
CONTENT_TYPE_TVSERIES_SEQUEL_NUM     = 47
CONTENT_TYPE_PLAYBACK_SPEED          = 'playback_speed'
CONTENT_TYPE_PLAYBACK_SPEED_NUM      = 48
CONTENT_TYPE_NUMERIC                 = 'number'
CONTENT_TYPE_NUMERIC_NUM             = 49
CONTENT_TYPE_APPLICATION             = 'application'
CONTENT_TYPE_APPLICATION_NUM         = 50
CONTENT_TYPE_SETTING                 = 'setting'
CONTENT_TYPE_SETTING_NUM             = 51
CONTENT_TYPE_MIXED_SEQUEL            = 'mixed_sequel'
CONTENT_TYPE_MIXED_SEQUEL_NUM        = 52
CONTENT_TYPE_OBSOLETETEAM            = 'obsoleteteam'
CONTENT_TYPE_OBSOLETETEAM_NUM        = 53
CONTENT_TYPE_OBSOLETETOURNAMENT      = 'obsoletetournament'
CONTENT_TYPE_OBSOLETETOURNAMENT_NUM  = 54
CONTENT_TYPE_DETAIL                  = 'detail'
CONTENT_TYPE_DETAIL_NUM              = 55
CONTENT_TYPE_DESCRIPTION             = 'description'
CONTENT_TYPE_DESCRIPTION_NUM         = 56
CONTENT_TYPE_KEYWORD                 = 'keyword'
CONTENT_TYPE_KEYWORD_NUM             = 57
CONTENT_TYPE_CREW                    = 'crew'
CONTENT_TYPE_CREW_NUM                = 58

# Content Xt Definitions
CONTENT_XT_PERSON                    = 'person'
CONTENT_XT_PERSON_NUM                = 1
CONTENT_XT_ACTOR                     = 'actor'
CONTENT_XT_ACTOR_NUM                 = 2
CONTENT_XT_DIRECTOR                  = 'director'
CONTENT_XT_DIRECTOR_NUM              = 3
CONTENT_XT_PRODUCER                  = 'producer'
CONTENT_XT_PRODUCER_NUM              = 4
CONTENT_XT_HOST                      = 'host'
CONTENT_XT_HOST_NUM                  = 5
CONTENT_XT_COMEDIAN                  = 'comedian'
CONTENT_XT_COMEDIAN_NUM              = 6
CONTENT_XT_PLAYER                    = 'player'
CONTENT_XT_PLAYER_NUM                = 7
CONTENT_XT_MODEL                     = 'model'
CONTENT_XT_MODEL_NUM                 = 8
CONTENT_XT_MUSIC_ARTIST              = 'musicartist'
CONTENT_XT_MUSIC_ARTIST_NUM          = 9
CONTENT_XT_MUSIC_BAND                = 'musicband'
CONTENT_XT_MUSIC_BAND_NUM            = 10
CONTENT_XT_DUO                       = 'duo'
CONTENT_XT_DUO_NUM                   = 11
CONTENT_XT_GENRE                     = 'genre'
CONTENT_XT_GENRE_NUM                 = 12
CONTENT_XT_SUBGENRE                  = 'subgenre'
CONTENT_XT_SUBGENRE_NUM              = 13
CONTENT_XT_MOOD                      = 'mood'
CONTENT_XT_MOOD_NUM                  = 14
CONTENT_XT_CLASSIC_GAME              = 'classicgame'
CONTENT_XT_CLASSIC_GAME_NUM          = 15
CONTENT_XT_PHRASE                    = 'phrase'
CONTENT_XT_PHRASE_NUM                = 16
CONTENT_XT_CONCEPT                   = 'concept'
CONTENT_XT_CONCEPT_NUM               = 17
CONTENT_XT_WIKI_CONCEPT              = 'wikiconcept'
CONTENT_XT_WIKI_CONCEPT_NUM          = 18
CONTENT_XT_PRIZE                     = 'prize'
CONTENT_XT_PRIZE_NUM                 = 19
CONTENT_XT_INSTRUMENT                = 'instrument'
CONTENT_XT_INSTRUMENT_NUM            = 20
CONTENT_XT_THEME                     = 'theme'
CONTENT_XT_THEME_NUM                 = 21
CONTENT_XT_PLAYING_STYLE             = 'playingstyle'
CONTENT_XT_PLAYING_STYLE_NUM         = 22
CONTENT_XT_VOCAL_TYPE                = 'vocaltype'
CONTENT_XT_VOCAL_TYPE_NUM            = 23
CONTENT_XT_TONE                      = 'tone'
CONTENT_XT_TONE_NUM                  = 24
CONTENT_XT_SONG_TYPE                 = 'songtype'
CONTENT_XT_SONG_TYPE_NUM             = 25
CONTENT_XT_SINGLE                    = 'single'
CONTENT_XT_SINGLE_NUM                = 26
CONTENT_XT_DUMMPY_TO_BE_USED         = 'to_be_used'
CONTENT_XT_DUMMPY_TO_BE_USED_NUM     = 27
CONTENT_XT_AVAIL_FILTER              = 'avail_filter'
CONTENT_XT_AVAIL_FILTER_NUM          = 28
CONTENT_XT_AWARD_CATEGORY            = 'award_category'
CONTENT_XT_AWARD_CATEGORY_NUM        = 29
CONTENT_XT_FILTER                    = 'filter'
CONTENT_XT_FILTER_NUM                = 30
CONTENT_XT_RATING_FILTER             = 'rating_filter'
CONTENT_XT_RATING_FILTER_NUM         = 31
CONTENT_XT_RESULT_FILTER             = 'result_filter'
CONTENT_XT_RESULT_FILTER_NUM         = 32
CONTENT_XT_SORT_FILTER               = 'sort_filter'
CONTENT_XT_SORT_FILTER_NUM           = 33
CONTENT_XT_SPORTS_FILTER             = 'sports_filter'
CONTENT_XT_SPORTS_FILTER_NUM         = 34
CONTENT_XT_STAR_FILTER               = 'star_filter'
CONTENT_XT_STAR_FILTER_NUM           = 35
CONTENT_XT_TAG_FILTER                = 'tag_filter'
CONTENT_XT_TAG_FILTER_NUM            = 36
CONTENT_XT_GUEST                     = 'guest'
CONTENT_XT_GUEST_NUM                 = 37
CONTENT_XT_ROVI_UNBOUND              = 'rovi_unbound'
CONTENT_XT_ROVI_UNBOUND_NUM          = 38
CONTENT_XT_SIMILAR_ARTIST            = 'similar_artist'
CONTENT_XT_SIMILAR_ARTIST_NUM        = 39
CONTENT_XT_SIMILAR_COMPOSER          = 'similar_composer'
CONTENT_XT_SIMILAR_COMPOSER_NUM      = 40
CONTENT_XT_SIMILAR_SOUND             = 'similar_sound'
CONTENT_XT_SIMILAR_SOUND_NUM         = 41
CONTENT_XT_SIMILAR_ALBUM             = 'similar_album'
CONTENT_XT_SIMILAR_ALBUM_NUM         = 42
CONTENT_XT_INFLUENCED                = 'influenced'
CONTENT_XT_INFLUENCED_NUM            = 43
CONTENT_XT_INFLUENCED_BY             = 'influenced_by'
CONTENT_XT_INFLUENCED_BY_NUM         = 44
CONTENT_XT_MOVIE_TYPE_THEATRE        = 'film'
CONTENT_XT_MOVIE_TYPE_THEATRE_NUM    = 45
CONTENT_XT_MOVIE_TYPE_TV             = 'tvfilm'
CONTENT_XT_MOVIE_TYPE_TV_NUM         = 46
CONTENT_XT_MOVIE_TYPE_VIDEO          = 'directtovideo'
CONTENT_XT_MOVIE_TYPE_VIDEO_NUM      = 47
CONTENT_XT_MOVIE_TYPE_WEB            = 'directtoweb'
CONTENT_XT_MOVIE_TYPE_WEB_NUM        = 48
CONTENT_XT_MOVIE_TYPE_CABLE          = 'madeforcable'
CONTENT_XT_MOVIE_TYPE_CABLE_NUM      = 49
CONTENT_XT_PRIZE_NORMALIZED          = 'prize_normalized'
CONTENT_XT_PRIZE_NORMALIZED_NUM      = 50
CONTENT_XT_COMPANY                   = 'company'
CONTENT_XT_COMPANY_NUM               = 51
CONTENT_XT_OBSOLETE_TEAM             = 'obsoleteteam'
CONTENT_XT_OBSOLETE_TEAM_NUM         = 52
CONTENT_XT_SPORTS_PHRASE             = 'sports_phrase'
CONTENT_XT_SPORTS_PHRASE_NUM         = 52
CONTENT_XT_PRIVATE_AFFILIATION       = 'private_affiliation'
CONTENT_XT_PRIVATE_AFFILIATION_NUM   = 53
CONTENT_XT_CATCHUP_FILTER            = 'channel_catchup_filter'
CONTENT_XT_CATCHUP_FILTER_NUM        = 54
CONTENT_XT_SOURCE_FILTER             = 'source_filter'
CONTENT_XT_SOURCE_FILTER_NUM         = 55
CONTENT_XT_CHANNEL_GROUP             = 'channelgroup'
CONTENT_XT_CHANNEL_GROUP_NUM         = 56
CONTENT_XT_OBSOLETE_TOURNAMENT       = 'obsoletetournament'
CONTENT_XT_OBSOLETE_TOURNAMENT_NUM   = 57
CONTENT_XT_MIXED                     = 'mixed'
CONTENT_XT_MIXED_NUM                 = 58
CONTENT_XT_ORPHAN                    = 'orphan'
CONTENT_XT_ORPHAN_NUM                = 59
CONTENT_XT_EVENT                     = 'event'
CONTENT_XT_EVENT_NUM                 = 60
CONTENT_XT_SETTINGACTION             = 'settingaction'
CONTENT_XT_SETTINGACTION_NUM         = 61

PROGRAM_TYPES = [ CONTENT_TYPE_MOVIE, CONTENT_TYPE_TVSERIES, CONTENT_TYPE_EPISODE, CONTENT_TYPE_TVVIDEO ]

ROVI_TYPE_LIST = [
    CONTENT_TYPE_EPISODE, CONTENT_TYPE_MOVIE,  CONTENT_TYPE_TVSERIES,
    CONTENT_TYPE_TVVIDEO, CONTENT_TYPE_PERSON, CONTENT_TYPE_PC,
    CONTENT_TYPE_SPORT, CONTENT_TYPE_TEAM
]

FOLD_TYPE_LIST = [ CONTENT_TYPE_LANGUAGE, CONTENT_TYPE_GENRE, CONTENT_TYPE_FILTER,
                   CONTENT_TYPE_DECADE, CONTENT_TYPE_STADIUM, CONTENT_TYPE_PHRASE,
                   CONTENT_TYPE_TOURNAMENT, CONTENT_TYPE_TEAM, CONTENT_TYPE_SPORTS_GROUP,
                   CONTENT_TYPE_AWARD, CONTENT_TYPE_PERSON, CONTENT_TYPE_SPORT,
                   CONTENT_TYPE_REGION, CONTENT_TYPE_OAC, CONTENT_TYPE_PC
                 ]

INCREMENTAL_KG_TYPE_LIST = [
    CONTENT_TYPE_MOVIE,  CONTENT_TYPE_TVSERIES, CONTENT_TYPE_EPISODE,
    CONTENT_TYPE_PERSON, CONTENT_TYPE_ROLE,     CONTENT_TYPE_REGION,
    CONTENT_TYPE_GENRE,  CONTENT_TYPE_SPORT,    CONTENT_TYPE_LANGUAGE,
    CONTENT_TYPE_PC, CONTENT_TYPE_RATING, CONTENT_TYPE_TEAM, CONTENT_TYPE_TOURNAMENT
]

ACE_SOURCE_STRING      = 'ACE'
ACE_SOURCE_STRING_DEV  = 'ACE_DEV'
ACE_SOURCE_STRING_QA1  = 'ACE_QA1'
ACE_SOURCE_STRING_QA2  = 'ACE_QA2'
ACE_SOURCE_STRING_UAT  = 'ACE_UAT'
ACE_SOURCE_STRING_PROD = 'ACE_PROD'
ACE_SOURCE_STRING_SDR  = 'ACE_SDR'
ACE_SOURCE_STRING_SDR_DEV  = 'ACE_SDR_DEV'
ACE_SOURCE_STRING_SDR_PROD = 'ACE_SDR_PROD'
ACE_SOURCE_STRING_DAZN_DEV = 'ACE_DAZN_DEV'
MVPSPA_SOURCE_STRING   = 'MVPSPA'
TIMESTAMP_FORMAT       = '%Y%m%dT%H%M%S'
DATETIME_FORMAT        = '%Y-%m-%d %H:%M:%S'

NON_KG_SOURCE_DB_TUPLE = (ACE_SOURCE_STRING_DEV, ACE_SOURCE_STRING_QA1, ACE_SOURCE_STRING_QA2, ACE_SOURCE_STRING_UAT, ACE_SOURCE_STRING_PROD, ACE_SOURCE_STRING_SDR, MVPSPA_SOURCE_STRING, ACE_SOURCE_STRING_SDR_DEV, ACE_SOURCE_STRING_SDR_PROD, ACE_SOURCE_STRING_DAZN_DEV)

ATTR_TO_XT_HASH = {
    'Pa' : CONTENT_XT_ACTOR,
    'Di' : CONTENT_XT_DIRECTOR,
    'Pr' : CONTENT_XT_PRODUCER,
    'Ho' : CONTENT_XT_HOST,
    'Co' : CONTENT_XT_MUSIC_ARTIST,
    'Zc' : CONTENT_XT_GUEST
    }

SEQUEL_XT_MOVIE = 'movie'
SEQUEL_XT_TVSERIES = 'tvseries'

CREW_ATTR_LIST = ('Ca', 'Di', 'Pr', 'Ho', 'Co', 'Uc', 'Ic', 'Zc', 'Wr')
CREW_MERGE_ATTR_LIST = ('Pa', 'Di', 'Pr', 'Ho', 'Co', 'Uc', 'Ic', 'Zc', 'Wr')

ROLE_ATTR_LIST = ('Sk', 'Pa', 'Cg')

IN_FILE_SUFFIX     = 'in'
SORTED_FILE_SUFFIX = 'sorted'

ROVI_GENRE_PREFIX = 'RVG'

GID_PREFIX_PREFERENCE = list(VALID_WIKI_PARENT_GID_PREFIX)
GID_PREFIX_PREFERENCE.extend([ 'FRB', 'CHAF', 'G', 'CG', 'SG', 'AR', 'LFMSO', 'LFMAR', 'TEAM', 'TOU', 'PL', 'RV' ])

LANG_MAPPING = {

    'Arabic Generic': 'ARA',
    'Basque Generic': 'BAQ',
    'Bengali Generic': 'BEN',
    'Catalán Generic': 'CAT',
    'Czech Generic': 'CES',
    'Chinese-Simplified-Mandarin': 'ZHO',
    'Chinese-S-Mandarin-Generic': 'ZHO',
    'Chinese-T-Mandarin-Generic': 'ZHO',
    'Chinese-T-Cantonese-Generic': 'ZHO',
    'Croatian_Generic': 'HRV',
    'Danish Generic': 'DAN',
    'Dutch Generic': 'DUT',
    'Dutch - Belgium': 'DUT',
    'English - AU': 'ENG',
    'English - NA': 'ENG',
    'English - UK': 'ENG',
    'Estonian Generic': 'EST',
    'Finnish Generic': 'FIN',
    'French Generic': 'FRA',
    'French - Québec': 'FRA',
    'French Belgium': 'FRA',
    'French France': 'FRA',
    'Gallegan Generic': 'GLG',
    'German Generic': 'GER',
    'Greek Generic': 'ELL',
    'Gujarati Generic': 'GUJ',
    'Hindi Generic': 'HIN',
    'Hungarian Generic': 'HUN',
    'Icelandic Generic': 'ISL',
    'Irish Generic': 'GLE',
    'Italian Generic': 'ITA',
    'Japanese Generic': 'JPN',
    'Kannada Generic': 'KAN',
    'Luxembourgish Generic': 'LTZ',
    'Malay Generic': 'MSA',
    'Malayalam Generic': 'MAL',
    'Maltese Generic': 'MLT',
    'Norwegian Generic': 'NOR',
    'Polish Generic': 'POL',
    'Portuguese': 'POR',
    'Portuguese Generic': 'POR',
    'Portuguese - Brazil': 'POR',
    'Portuguese - Portugal': 'POR',
    'Russian-C Generic': 'RUS',
    'Russian Generic': 'RUS',
    'Scots Generic': 'SCO',
    'Scottish Gaelic Generic': 'GLA',
    'Serbian_Generic': 'SRP',
    'Slovak Generic': 'SLK',
    'Spanish': 'SPA',
    'Spanish Generic': 'SPA',
    'Spanish Spain': 'SPA',
    'Spanish USA': 'SPA',
    'Swedish Generic': 'SWE',
    'Tamil Generic': 'TAM',
    'Welsh Generic': 'WEL',
    'Tagalog Generic': 'TGL',
    'Turkish Generic': 'TUR',
    'Ukrainian Generic': 'UKR',
    }

ROVI_LANG_FILES_CODE_LIST = [
    'FRA', 'JPN', 'DAN', 'SWE', 'GER',
    'ITA', 'NOR', 'POL', 'DUT', 'RUS',
    'SPA', 'KOR', 'FIN', 'POR', 'ZHO',
    'MSA', 'CAT', 'ENG', 'TUR', 'UKR',
    'VIE', 'IND', 'HUN', 'RON', 'CES',
    'HRV', 'SRP'
]

COMMON_LANGUAGE_CODE_TO_REGIONAL_LANGUAGE_CODE = {
    'GER': 'DEU',
    'DUT': 'NLD',
}

ZL_MAP = {
    'DEU': 'GER',
    'NLD': 'DUT',
    'FRE': 'FRA',
}

# %s -> 3 letter iso lang code
ROVI_OTHER_LANGUAGE_FILE_PATTERN = "rovi_%s.data"

PERSON_OTHER_LANGUAGE_FILE_PATTERN = "crew_%s.data"

# in the fd field for rovi 2.0 id identifier
FD_ROVI_ID_NAME = "rovi_id_2.0"
FD_ROVI_GROUP_ID_NAME = "group_id"
ROVI_1_1_ID = 'rovi_id_1.1'
ROVI_CONTENT_ID = 'rovi_content_id'

# Popularity Ranges
POPULARITY_VT_RANGE = {
    CONTENT_TYPE_ALBUM: (0, 100),
    CONTENT_TYPE_AWARD: (0, 300),
    CONTENT_TYPE_CHANNEL: (0, 1500),
    CONTENT_TYPE_CHANNEL_AFFILIATION: (0, 1500),
    CONTENT_TYPE_EPISODE: (0, 200),
    CONTENT_TYPE_GAME: (0, 80),
    CONTENT_TYPE_GENRE: (0, 200),
    CONTENT_TYPE_LANGUAGE: (0, 200),
    CONTENT_TYPE_MOVIE: (0, 550),
    CONTENT_TYPE_PERSON: (0, 600),
    CONTENT_TYPE_ROLE: (0, 600),
    CONTENT_TYPE_SEQUEL: (0, 850),
    CONTENT_TYPE_SONG: (0, 200),
    CONTENT_TYPE_SPORT: (0, 600),
    CONTENT_TYPE_SPORTS_GROUP: (0, 250),
    CONTENT_TYPE_STADIUM: (0, 500),
    CONTENT_TYPE_TEAM: (0, 400),
    CONTENT_TYPE_TOURNAMENT: (0, 500),
    CONTENT_TYPE_TVSERIES: (0, 850),
}

SEED_IGS_SOURCE_STRING = 'seed_igs'

#TYPE-WISE FIELDS LIST
FIRST_FIELDS          = [ 'Gi', 'So', 'Vt' ]
TITLE_FIELDS          = [ 'Ti', 'Ep', 'Ak', 'Ae', 'Va', 'Ik', 'Bs', 'Za' ]
COMMON_FIELDS         = FIRST_FIELDS + [ 'Xt', 'Mi', 'Pi' ] + TITLE_FIELDS
# Field per Type
MOVIE_FIELDS      = [ 'Ac', 'Ae', 'Ak', 'Aw', 'Bs', 'Cb', 'Cl', 'Co', 'Di', 'Dl', 'Du', 'Ep', 'Ge', 'Gg', 'Gi', 'Ic', 'Ig', 'Ll', 'Ml', 'Od', 'Oy', 'Pa', 'Pc', 'Pr', 'Ra', 'Ry', 'Sq', 'Tg', 'Ti', 'Tl', 'Uc', 'Va', 'Vt', 'Wr', 'Xt', 'Zc', 'Zi' ]
TVSERIES_FIELDS   = [ 'Ac', 'Ae', 'Af', 'Ak', 'Aw', 'Bs', 'Cb', 'Cl', 'Co', 'Di', 'Dl', 'Du', 'Ep', 'Ed', 'Ge', 'Gg', 'Gi', 'Ho', 'Ic', 'Ig', 'Ll', 'Ml', 'Od', 'Oy', 'Pa', 'Pc', 'Pr', 'Ra', 'Ry', 'Sq', 'Tg', 'Ti', 'Tl', 'Uc', 'Va', 'Vt', 'Wr', 'Xt', 'Zc', 'Zi' ]
EPISODE_FIELDS    = [ 'Ac', 'Ae', 'Af', 'Ak', 'Aw', 'Bs', 'Cb', 'Cl', 'Co', 'Di', 'Dl', 'Du', 'Ed', 'En', 'Ep', 'Ge', 'Gg', 'Gi', 'Ho', 'Ic', 'Ig', 'Ll', 'Ml', 'Od', 'Oy', 'Pa', 'Pc', 'Pn', 'Pr', 'Ra', 'Ry', 'Sq', 'Sn', 'Tg', 'Ti', 'Tl', 'Tn', 'Uc', 'Va', 'Vt', 'Wr', 'Xt', 'Zc', 'Zi' ]
AWARD_FIELDS      = [ 'Ak', 'Bs', 'Cl', 'Gi', 'Ik', 'Ml', 'Nc', 'Pi', 'Ti', 'Tl', 'Va', 'Vt', 'Xt', 'Zi' ]
FILTER_FIELDS     = [ 'Ak', 'Gi', 'Tg', 'Ti', 'Vt', 'Xt' ]
CHANNEL_AF_FIELDS = [ 'Ak', 'Bp', 'Dp', 'Fi', 'Gi', 'Ow', 'Pi', 'Tg', 'Ti', 'Vt', 'Xt' ]
CHANNEL_FILEDS    = [ 'Af', 'Ag', 'Ak', 'Bp', 'Cl', 'Cs', 'Ct', 'Ge', 'Gi', 'Ll', 'Ng', 'Og', 'Ow', 'Pi', 'Rg', 'Sc', 'Sl', 'St', 'Tg', 'Ti', 'Vt', 'Xt' ]
DECADE_FIELDS     = [ 'Ak', 'Gi', 'Od', 'Ry', 'Ti', 'Vt' ]
DETAIL_FIELDS     = [ 'Gi', 'Ti', 'Vt', 'Xt' ]
GENRE_FIELDS      = [ 'Ak', 'Bs', 'Cl', 'Gi', 'Ik', 'Mp', 'Np', 'Pi', 'Tg', 'Ti', 'Um', 'Va', 'Vt', 'Xt' ]
OAC_FIELDS        = [ 'Gi', 'Ti', 'Va', 'Vt' ]
PC_FIELDS         = [ 'Ak', 'Aw', 'Bs', 'Cl', 'Gi', 'Ik', 'Od', 'Ti', 'Tl', 'Va', 'Vt', 'Xt', 'Zi' ]
RATING_FIELDS     = [ 'Ak', 'Cl', 'Gi', 'Rg', 'Rk', 'Rv', 'Ti', 'Va', 'Vt', 'Xt' ]
REGION_FIELDS     = [ 'Ak', 'Cl', 'Gi', 'Ik', 'Ti', 'Va', 'Vt', 'Xt', 'Zi' ]
LANGUAGE_FIELDS   = [ 'Ak', 'Bs', 'Cl', 'Gi', 'Ik', 'Od', 'Pi', 'Ti', 'Va', 'Vt', 'Zi' ]
ROLE_FIELDS       = [ 'Ak', 'Cg', 'Gi', 'Oc', 'Pa', 'Tg', 'Ti', 'Vt' ]
SEQUEL_FIELDS     = [ 'Fi', 'Gi', 'Np', 'Pa', 'Ti', 'Vt', 'Wg', 'Xt' ]
SPORT_FIELDS      = [ 'Ak', 'Cl', 'Gi', 'Ik', 'Pi', 'Ti', 'Tl', 'Va', 'Vt', 'Zi' ]
GROUP_FIELDS      = [ 'Ak', 'Bs', 'Cl', 'Gi', 'Ik', 'Ll', 'Pi', 'Rg', 'Sp', 'Tg', 'Ti', 'To', 'Va', 'Vt', 'Xt', 'Zi' ]
STADIUM_FIELDS    = [ 'Ak', 'Bs', 'Cl', 'Ct', 'Gi', 'Ik', 'Rg', 'Sp', 'St', 'Ti', 'Va', 'Vt', 'Zi' ]
TEAM_FIELDS       = [ 'Af', 'Ak', 'Bs', 'Cl', 'Cs', 'Dt', 'Gi', 'Ik', 'Lo', 'Rg', 'Rt', 'Sg', 'Sm', 'Sp', 'Sx', 'Tg', 'Ti', 'Tl', 'To', 'Va', 'Vt', 'Xt', 'Zi' ]
TOURNAMENT_FIELDS = [ 'Af', 'Ak', 'Bs', 'Cl', 'Gi', 'Ik', 'Pi', 'Pt', 'Rd', 'Rg', 'Sm', 'Sp', 'Sx', 'Te', 'Tg', 'Ti', 'Tl', 'Va', 'Vt', 'Xt', 'Za', 'Zi' ]
PERSON_FIELDS     = [ 'Af', 'Ak', 'Aw', 'Bd', 'Bm', 'Bs', 'Cl', 'Do', 'Dt', 'Ge', 'Gi', 'Ik', 'Ml', 'Oc', 'Rd', 'Rg', 'Ro', 'Rt', 'Sg', 'Sp', 'Sx', 'Te', 'Tg', 'Ti', 'Tl', 'To', 'Va', 'Vt', 'Xt', 'Zi' ]
PHRASE_FIELDS     = [ 'Ak', 'Bs', 'Gi', 'Ti', 'Tl', 'Va', 'Vt', 'Xt', 'Zi' ]
DESC_FIELDS       = [ 'De', 'Gi', 'Vt', 'Xt', 'Zl' ]
KWD_FIELDS        = [ 'Gi', 'Ke', 'Vt', 'Xt', 'Zl' ]
DEFAULT_FIELDS    = PHRASE_FIELDS

TYPE_FIELD_DICT   = { 'movie'              : MOVIE_FIELDS,
                      'tvseries'           : TVSERIES_FIELDS,
                      'episode'            : EPISODE_FIELDS,
                      'award'              : AWARD_FIELDS,
                      'filter'             : FILTER_FIELDS,
                      'channelaffiliation' : CHANNEL_AF_FIELDS,
                      'channel'            : CHANNEL_FILEDS,
                      'decade'             : DECADE_FIELDS,
                      'detail'             : DETAIL_FIELDS,
                      'genre'              : GENRE_FIELDS,
                      'originalchannel'    : OAC_FIELDS,
                      'productionhouse'    : PC_FIELDS,
                      'rating'             : RATING_FIELDS,
                      'region'             : REGION_FIELDS,
                      'language'           : LANGUAGE_FIELDS,
                      'role'               : ROLE_FIELDS,
                      'sequel'             : SEQUEL_FIELDS,
                      'sport'              : SPORT_FIELDS,
                      'sportsgroup'        : GROUP_FIELDS,
                      'stadium'            : STADIUM_FIELDS,
                      'team'               : TEAM_FIELDS,
                      'tournament'         : TOURNAMENT_FIELDS,
                      'person'             : PERSON_FIELDS,
                      'phrase'             : PHRASE_FIELDS
                    }
