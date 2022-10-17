#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#$Id: lang_utils.py,v 1.41 2017/09/11 13:29:11 manoj Exp $

import sys
import string
from itertools import chain
from operator import itemgetter
from collections import defaultdict
from .data_constants import *

perl_file_header = '''
#This file is auto generated from pratham/src/Common/utils/lang_utils.py
'''
perl_translate_string = '''
my %translate_string=(
    0x21,'!',           0x22,'"',           0x23,'#',           0x24,'$',           0x25,'%',
    0x26,'&',           0x27,"'",           0x28,'(',           0x29,')',           0x2a,'*',
    0x2b,'+',           0x2c,',',           0x2d,'-',           0x2e,'.',           0x2f,'/',
    0x30,'0',           0x31,'1',           0x32,'2',           0x33,'3',           0x34,'4',
    0x35,'5',           0x36,'6',           0x37,'7',           0x38,'8',           0x39,'9',
    0x3a,':',           0x3b,';',           0x3c,'<',           0x3d,'=',           0x3e,'>',
    0x3f,'?',           0x40,'@',           0x41,'A',           0x42,'B',           0x43,'C',
    0x44,'D',           0x45,'E',           0x46,'F',           0x47,'G',           0x48,'H',
    0x49,'I',           0x4a,'J',           0x4b,'K',           0x4c,'L',           0x4d,'M',
    0x4e,'N',           0x4f,'O',           0x50,'P',           0x51,'Q',           0x52,'R',
    0x53,'S',           0x54,'T',           0x55,'U',           0x56,'V',           0x57,'W',
    0x58,'X',           0x59,'Y',           0x5a,'Z',           0x5b,'[',           0x5c,'\\\\',
    0x5d,']',           0x5f,'_',           0x61,'a',           0x62,'b',           0x63,'c',
    0x64,'d',           0x65,'e',           0x66,'f',           0x67,'g',           0x68,'h',
    0x69,'i',           0x6a,'j',           0x6b,'k',           0x6c,'l',           0x6d,'m',
    0x6e,'n',           0x6f,'o',           0x70,'p',           0x71,'q',           0x72,'r',
    0x73,'s',           0x74,'t',           0x75,'u',           0x76,'v',           0x77,'w',
    0x78,'x',           0x79,'y',           0x7a,'z',           0x7c,'|',           0x7e,'~',
    0xa1,'!',           0xa2,'{cent}',      0xa3,'{pound}',     0xa4,'{currency}',  0xa5,'{yen}',
    0xa6,'|',           0xa7,'{section}',   0xa8,'{umlaut}',    0xa9,'{C}',         0xaa,'{^a}',
    0xab,'<<',          0xac,'{not}',       0xad,'-',           0xae,'{R}',         0xaf,'_',
    0xb0,'{degrees}',   0xb1,'{+/-}',       0xb2,'{^2}',        0xb3,'{^3}',        0xb4,"'",
    0xb5,'u',     0xb6,'{paragraph}', 0xb7,' ',           0xb8,'{cedilla}',   0xb9,'{^1}',
    0xba,'{^o}',        0xbb,'>>',          0xbc,'{1/4}',       0xbd,'{1/2}',       0xbe,'{3/4}',
    0xbf,'?',           0xc0,'A',           0xc1,'A',           0xc2,'A',           0xc3,'A',
    0xc4,'A',           0xc5,'A',           0xc6,'Ae',          0xc7,'C',           0xc8,'E',
    0xc9,'E',           0xca,'E',           0xcb,'E',           0xcc,'I',           0xcd,'I',
    0xce,'I',           0xcf,'I',           0xd0,'Th',          0xd1,'N',           0xd2,'O',
    0xd3,'O',           0xd4,'O',           0xd5,'O',           0xd6,'O',           0xd7,'x',
    0xd8,'O',           0xd9,'U',           0xda,'U',           0xdb,'U',           0xdc,'U',
    0xdd,'Y',           0xde,'th',          0xdf,'ss',          0xe0,'a',           0xe1,'a',
    0xe2,'a',           0xe3,'a',           0xe4,'a',           0xe5,'a',           0xe6,'ae',
    0xe7,'c',           0xe8,'e',           0xe9,'e',           0xea,'e',           0xeb,'e',
    0xec,'i',           0xed,'i',           0xee,'i',           0xef,'i',           0xf0,'th',
    0xf1,'n',           0xf2,'o',           0xf3,'o',           0xf4,'o',           0xf5,'o',
    0xf6,'o',           0xf7,'/',           0xf8,'o',           0xf9,'u',           0xfa,'u',
    0xfb,'u',           0xfc,'u',           0xfd,'y',           0xfe,'th',          0xff,'y',
    0x100,'A',          0x101,'a',          0x112,'E',          0x113,'e',          0x12a,'I',
    0x12b,'i',          0x14c,'O',          0x14d,'o',          0x16a,'U',          0x16b,'u',
    0x232,'Y',          0x233,'y',          0x1e20,'G',         0x1e21,'g',
);
'''

perl_functions = '''
sub is_in_set{
    my ($char_map_set, $char_ord) = @_;
    return defined $char_map_set->{$char_ord};
}

sub is_in_range{
    my ($char_map_range_list, $char_ord) = @_;
    foreach my $each_range (@$char_map_range_list){
        my ($lower_lim, $upper_lim) = @$each_range;
        if($char_ord ge $lower_lim and $char_ord le $upper_lim)
        {
            return 1;
        }
    }
    return 0;
}

sub char_map{
    my($chartranslate, $char_ord, $is_in) = @_;
    if ($is_in->($char_ord))
    {
        return $char_ord;
    }
    else
    {
        return ' ';
    }
}

sub lang_clean{
    my ($this, $lang_obj_ref, $val_string, $string_transform) = @_;
    my $translate_regex = $lang_obj_ref->{"translate_regex"};
    my $chartranslate = $lang_obj_ref->{"chartranslate"};
    my $new_val_string = '';

    for my $each (split "", $val_string){
        if (defined $chartranslate->{$each}){
            $new_val_string = $new_val_string . $chartranslate->{$each};
        }
        else{
            $new_val_string = $new_val_string . $each;
        }
    }
    $val_string = lc $new_val_string;
    if(defined $string_transform)
    {
        $val_string = $string_transform->($val_string);
    }
    if (defined $lang_obj_ref->{"charset_regex"})
    {
        $val_string =~ s/$lang_obj_ref->{"charset_regex"}/ /g;
    }
    else
    {
        my $char_map_func = $lang_obj_ref->{"char_map"};
        $val_string = join "", map { $char_map_func->($_) } split("", $val_string);
    }
    return $val_string;
}

sub create_char_map_set_obj {
    my ($charset, $chartranslate) = @_;
    my $translate_regex = join '|', map { quotemeta($_ )} sort keys %{$chartranslate};
    my $charset_regex = '[^' . (join '', map {quotemeta($_) } sort keys %{$charset} ). ']';
    my $is_in = sub {
        my ($char_ord) = @_;
        return is_in_set($charset, $char_ord);
    };
    my $char_map_for_set = sub {
        my$char_ord = shift;
        return char_map($chartranslate, $char_ord, $is_in);
    };
    my %lang_obj = (
        "is_in" => $is_in,
        "char_map" => $char_map_for_set,
        "chartranslate" => $chartranslate,
        "charset" => $charset,
        "translate_regex" => $translate_regex,
        "charset_regex" => $charset_regex );
    return \%lang_obj;
}

sub create_char_map_range_obj {
    my ($charrange, $chartranslate) = @_;
    my $translate_regex = join '|', map { quotemeta($_ )} sort keys %{$chartranslate};
    my $is_in = sub {
        my ($char_ord) = @_;
        return is_in_range($charrange, $char_ord);
    };
    my $char_map_for_range = sub {
        my$char_ord = shift;
        return char_map($chartranslate, $char_ord, $is_in);
    };
    my %lang_obj = (
        "is_in" => $is_in,
        "char_map" => $char_map_for_range,
        "chartranslate" => $chartranslate,
        "translate_regex" => $translate_regex,
        "charrange" => $charrange);
    return \%lang_obj;
}


'''
latin_ascii_xlate_ext = defaultdict(lambda: ' ')

tmp_latin_ascii_xlate = {
    0xc0: 'A', 0xc1: 'A', 0xc2: 'A', 0xc3: 'A', 0xc4: 'A', 0xc5: 'A', 0x0100: 'A',
        0xc6: 'Ae', 0xc7: 'C',
        0xc8: 'E', 0xc9: 'E', 0xca: 'E', 0xcb: 'E', 0x0112: 'E',
        0x1e20: 'G',
        0xc7: 'C', 0xcc: 'I', 0xcd: 'I', 0xce: 'I', 0xcf: 'I', 0x012a: 'I',
        0xd0: 'Th', 0xd1: 'N',
        0xd2: 'O', 0xd3: 'O', 0xd4: 'O', 0xd5: 'O', 0xd6: 'O', 0xd8: 'O', 0x014c: 'O',
        0xd9: 'U', 0xda: 'U', 0xdb: 'U', 0xdc: 'U', 0x016a: 'U',
        0xdd: 'Y', 0x0232: 'Y', 0xde: 'th', 0xdf: 'ss',
        0xe0: 'a', 0xe1: 'a', 0xe2: 'a', 0xe3: 'a', 0xe4: 'a', 0xe5: 'a', 0x0101: 'a',
        0xe6: 'ae', 0xe7: 'c',
        0xe8: 'e', 0xe9: 'e', 0xea: 'e', 0xeb: 'e', 0x0113: 'e',
        0x1e21: 'g',
        0xec: 'i', 0xed: 'i', 0xee: 'i', 0xef: 'i', 0x012b: 'i',
        0xf0: 'th', 0xf1: 'n',
        0xf2: 'o', 0xf3: 'o', 0xf4: 'o', 0xf5: 'o', 0xf6: 'o', 0xf8: 'o', 0x014d: 'o',
        0xf9: 'u', 0xfa: 'u', 0xfb: 'u', 0xfc: 'u', 0x016b: 'u',
        0xfd: 'y', 0xfe: 'th', 0xff: 'y', 0x0233: 'y',
        0xa1: '!', 0xa2: '{cent}', 0xa3: '{pound}', 0xa4: '{currency}',
        0xa5: '{yen}', 0xa6: '|', 0xa7: '{section}', 0xa8: '{umlaut}',
        0xa9: '{C}', 0xaa: '{^a}', 0xab: '<<', 0xac: '{not}',
        0xad: '-', 0xae: '{R}', 0xaf: '_', 0xb0: '{degrees}',
        0xb1: '{+/-}', 0xb2: '{^2}', 0xb3: '{^3}', 0xb4: "'",
        0xb5: 'u', 0xb6: '{paragraph}', 0xb7: ' ', 0xb8: '{cedilla}',
        0xb9: '{^1}', 0xba: '{^o}', 0xbb: '>>',
        0xbc: '{1/4}', 0xbd: '{1/2}', 0xbe: '{3/4}', 0xbf: '?', 0xd7: 'x', 0xf7: '/',
        0xdc: 'U', 0xd6: 'O',
        0x102: 'A', 0x103: 'a', 0x106: 'C', 0x107: 'c', 0x116: 'E', 0x117: 'e',
        0x11e: 'G', 0x11f: 'g',
        0x131: 'i', 0x110: 'D', 0x126: 'H', 0x134: 'J', 0xff: 'ye', 0x14a: 'N',
        0x166: 'T', 0x11b: 'e', 0x17d: 'Z', 0x111: 'd', 0x17f: 'f', 0x127: 'h', 0x152: 'OE',
        0x135: 'j', 0x142: 'l', 0x14b: 'n', 0x138: 'k', 0x17e: 'z', 0x133: 'ij', 0x2122: 'TM',
        0x167: 't', 0x104: 'A', 0x10c: 'C', 0x11a: 'E', 0x122: 'G', 0x136: 'K', 0x150: 'O',
        0x141: 'L', 0x132: 'IJ', 0x172: 'U', 0x174: 'W', 0x178: 'Y', 0x159: 'r', 0x105: 'a',
        0x10d: 'c', 0x123: 'g', 0x153: 'oe', 0x151: 'o', 0x158: 'R', 0x15e: 'S', 0x15f: 's',
        0X173: 'u', 0x175: 'w', 0x177: 'y', 0x17b: 'Z', 0x17c: 'z',
        0x130: 'I', 0x160: 'S', 0x161: 's', 0x192: 'f', 0x218: 'S', 0x219: 's',
        0x21a: 'T', 0x21b: 't', 0x1ef2: 'Y', 0x1ef3: 'y',
        0x2122: 'tm', 0x2019: '\'',
        0xde: 'T',
        0xfe: 't',
        0x1eae: 'A',
        0x1eaf: 'a',
        0x1eb0: 'A',
        0x1eb1: 'a',
        0x1eb2: 'A',
        0x1eb3: 'a',
        0x1eb4: 'A',
        0x1eb5: 'a',
        0x1eb6: 'A',
        0x1eb7: 'a',
        0x114: 'E',
        0x115: 'e',
        0x12c: 'I',
        0x12d: 'i',
        0x14e: 'O',
        0x14f: 'o',
        0x16c: 'U',
        0x16d: 'u',
        0x20a: 'I',
        0x20b: 'i',
        0x206: 'E',
        0x207: 'e',
        0x202: 'A',
        0x203: 'a',
        0x212: 'R',
        0x20e: 'O',
        0x20f: 'o',
        0x213: 'r',
        0x216: 'U',
        0x217: 'u',
        0x1e1c: 'E',
        0x1e1d: 'e',
        0x1e2a: 'H',
        0x1e2b: 'h',
        0x108: 'C',
        0x109: 'c',
        0x10a: 'C',
        0x10b: 'c',
        0x10e: 'D',
        0x10f: 'd',
        0x118: 'E',
        0x119: 'e',
        0x11c: 'G',
        0x11d: 'g',
        0x120: 'G',
        0x121: 'g',
        0x124: 'H',
        0x125: 'h',
        0x128: 'I',
        0x129: 'i',
        0x12e: 'I',
        0x12f: 'i',
        0x137: 'k',
        0x139: 'A',
        0x13a: 'l',
        0x13b: 'L',
        0x13c: 'l',
        0x13d: 'L',
        0x13e: 'l',
        0x13f: 'L',
        0x140: 'l',
        0x143: 'N',
        0x144: 'n',
        0x145: 'N',
        0x146: 'n',
        0x147: 'N',
        0x148: 'n',
        0x154: 'R',
        0x155: 'r',
        0x156: 'R',
        0x157: 'r',
        0x15a: 'S',
        0x15b: 's',
        0x15c: 'S',
        0x15d: 's',
        0x162: 'T',
        0x163: 't',
        0x164: 'T',
        0x165: 't',
        0x168: 'U',
        0x169: 'u',
        0x16e: 'U',
        0x16f: 'u',
        0x170: 'U',
        0x171: 'u',
        0x176: 'Y',
        0x179: 'Z',
        0x17a: 'z',
        0x180: 'b',
        0x181: 'B',
        0x182: 'B',
        0x183: 'b',
        0x187: 'C',
        0x188: 'c',
        0x189: 'D',
        0x18a: 'D',
        0x191: 'F',
        0x193: 'G',
        0x197: 'I',
        0x198: 'K',
        0x199: 'k',
        0x19a: 'l',
        0x19d: 'N',
        0x19e: 'n',
        0x19f: 'O',
        0x1a0: 'O',
        0x1a1: 'o',
        0x1a4: 'P',
        0x1a5: 'p',
        0x1ab: 't',
        0x1ac: 'T',
        0x1ad: 't',
        0x1ae: 'T',
        0x1af: 'U',
        0x1b0: 'u',
        0x1b2: 'V',
        0x1b3: 'Y',
        0x1b4: 'y',
        0x1b5: 'Z',
        0x1b6: 'z',
        0x1ba: 'z',
        0x1cd: 'A',
        0x1ce: 'a',
        0x1cf: 'I',
        0x1d0: 'i',
        0x1d1: 'O',
        0x1d2: 'o',
        0x1d3: 'U',
        0x1d4: 'u',
        0x1e2: 'AE',
        0x1e3: 'ae',
        0x1fc: 'AE',
        0x1fd: 'ae',
        0x1e4: 'G',
        0x1e5: 'g',
        0x1e6: 'G',
        0x1e7: 'g',
        0x1e8: 'K',
        0x1e9: 'k',
        0x1ea: 'O',
        0x1eb: 'o',
        0x1ee: 'Z',
        0x1ef: 'z',
        0x1f0: 'j',
        0x1f4: 'G',
        0x1f5: 'g',
        0x1f8: 'N',
        0x1f9: 'n',
        0x1fa: 'A',
        0x1fb: 'a',
        0x1fe: 'O',
        0x1ff: 'o',
        0x200: 'A',
        0x201: 'a',
        0x204: 'E',
        0x205: 'e',
        0x208: 'I',
        0x209: 'i',
        0x20c: 'O',
        0x20d: 'o',
        0x210: 'R',
        0x211: 'r',
        0x214: 'U',
        0x215: 'u',
        0x1d5: 'U',
        0x1d6: 'u',
        0x1d7: 'U',
        0x1d8: 'u',
        0x1d9: 'U',
        0x1da: 'u',
        0x1db: 'U',
        0x1dc: 'u',
        0x1de: 'A',
        0x1df: 'a',
        0x1e0: 'A',
        0x1e1: 'a',
        0x1ec: 'O',
        0x1ed: 'o',
        0x21e: 'H',
        0x21f: 'h',
        0x220: 'N',
        0x22a: 'O',
        0x22b: 'o',
        0x22c: 'O',
        0x22d: 'o',
        0x230: 'O',
        0x231: 'o',
        0x221: 'd',
        0x224: 'Z',
        0x225: 'z',
        0x226: 'A',
        0x227: 'a',
        0x228: 'E',
        0x229: 'e',
        0x22e: 'O',
        0x22f: 'o',
        0x23a: 'A',
        0x23b: 'C',
        0x23c: 'c',
        0x23e: 'T',
        0x243: 'B',
        0x246: 'E',
        0x247: 'e',
        0x248: 'J',
        0x249: 'j',
        0x24c: 'R',
        0x24d: 'r',
        0x24e: 'Y',
        0x24f: 'y',
        0x234: 'l',
        0x235: 'n',
        0x236: 't',
        0x237: 'j',
        0x23d: 'L',
        0x244: 'U',
        0x24a: 'Q',
        0x24b: 'q',
        0x253: 'b',
        0x255: 'c',
        0x256: 'd',
        0x257: 'd',
        0x25f: 'j',
        0x260: 'g',
        0x268: 'i',
        0x26b: 'l',
        0x26c: 'l',
        0x26d: 'l',
        0x271: 'm',
        0x272: 'n',
        0x273: 'n',
        0x275: 'o',
        0x27c: 'r',
        0x27d: 'r',
        0x27e: 'r',
        0x282: 's',
        0x284: 'j',
        0x288: 't',
        0x289: 'u',
        0x28b: 'v',
        0x28f: 'y',
        0x290: 'z',
        0x291: 'z',
        0x29d: 'j',
        0x2a0: 'q',
        0x1d6c: 'b',
        0x1d6d: 'd',
        0x1d6e: 'f',
        0x1d72: 'r',
        0x1d73: 'r',
        0x1d75: 't',
        0x1e00: 'A',
        0x1e01: 'a',
        0x1e02: 'B',
        0x1e03: 'b',
        0x1e04: 'B',
        0x1e05: 'b',
        0x1e06: 'B',
        0x1e07: 'b',
        0x1e08: 'C',
        0x1e09: 'c',
        0x1e0a: 'D',
        0x1e0b: 'd',
        0x1e0c: 'D',
        0x1e0d: 'd',
        0x1e0e: 'D',
        0x1e0f: 'd',
        0x1e10: 'D',
        0x1e11: 'd',
        0x1e12: 'D',
        0x1e13: 'd',
        0x1e14: 'E',
        0x1e15: 'e',
        0x1e16: 'E',
        0x1e17: 'e',
        0x1e18: 'E',
        0x1e19: 'e',
        0x1e1a: 'E',
        0x1e1b: 'e',
        0x1e1e: 'F',
        0x1e1f: 'f',
        0x1e22: 'H',
        0x1e23: 'h',
        0x1e24: 'H',
        0x1e25: 'h',
        0x1e26: 'H',
        0x1e27: 'h',
        0x1e28: 'H',
        0x1e29: 'h',
        0x1e2c: 'I',
        0x1e2d: 'i',
        0x1e2e: 'I',
        0x1e2f: 'i',
        0x1e30: 'K',
        0x1e31: 'k',
        0x1e32: 'K',
        0x1e33: 'k',
        0x1e34: 'K',
        0x1e35: 'k',
        0x1e36: 'L',
        0x1e37: 'l',
        0x1e38: 'L',
        0x1e39: 'l',
        0x1e3a: 'L',
        0x1e3b: 'l',
        0x1e3c: 'L',
        0x1e3d: 'l',
        0x1e3e: 'M',
        0x1e3f: 'm',
        0x1e40: 'M',
        0x1e41: 'm',
        0x1e42: 'M',
        0x1e43: 'm',
        0x1e44: 'N',
        0x1e45: 'n',
        0x1e46: 'N',
        0x1e47: 'n',
        0x1e48: 'N',
        0x1e49: 'n',
        0x1e4a: 'N',
        0x1e4b: 'n',
        0x1e4c: 'O',
        0x1e4d: 'o',
        0x1e4e: 'O',
        0x1e4f: 'o',
        0x1e50: 'O',
        0x1e51: 'o',
        0x1e52: 'O',
        0x1e53: 'o',
        0x1e54: 'P',
        0x1e55: 'p',
        0x1e56: 'P',
        0x1e57: 'p',
        0x1e58: 'R',
        0x1e59: 'r',
        0x1e5a: 'R',
        0x1e5b: 'r',
        0x1e5c: 'R',
        0x1e5d: 'r',
        0x1e5e: 'R',
        0x1e5f: 'r',
        0x1e60: 'S',
        0x1e61: 's',
        0x1e62: 'S',
        0x1e63: 's',
        0x1e64: 'S',
        0x1e65: 's',
        0x1e66: 'S',
        0x1e67: 's',
        0x1e68: 'S',
        0x1e69: 's',
        0x1e6a: 'T',
        0x1e6b: 't',
        0x1e6c: 'T',
        0x1e6d: 't',
        0x1e6e: 'T',
        0x1e6f: 't',
        0x1e70: 'T',
        0x1e71: 't',
        0x1e72: 'U',
        0x1e73: 'u',
        0x1e74: 'U',
        0x1e75: 'u',
        0x1e76: 'U',
        0x1e77: 'u',
        0x1e78: 'U',
        0x1e79: 'u',
        0x1e7a: 'U',
        0x1e7b: 'u',
        0x1e7c: 'V',
        0x1e7d: 'v',
        0x1e7e: 'V',
        0x1e7f: 'v',
        0x1e80: 'W',
        0x1e81: 'w',
        0x1e82: 'W',
        0x1e83: 'w',
        0x1e84: 'W',
        0x1e85: 'w',
        0x1e86: 'W',
        0x1e87: 'w',
        0x1e88: 'W',
        0x1e89: 'w',
        0x1e8a: 'X',
        0x1e8b: 'x',
        0x1e8c: 'X',
        0x1e8d: 'x',
        0x1e8e: 'Y',
        0x1e8f: 'y',
        0x1e90: 'Z',
        0x1e91: 'z',
        0x1e92: 'Z',
        0x1e93: 'z',
        0x1e94: 'Z',
        0x1e95: 'z',
        0x1e96: 'h',
        0x1e97: 't',
        0x1e98: 'w',
        0x1e99: 'y',
        0x1e9a: 'a',
        0x1e9b: 's',
        0x1ea0: 'A',
        0x1ea1: 'a',
        0x1ea2: 'A',
        0x1ea3: 'a',
        0x1ea4: 'A',
        0x1ea5: 'a',
        0x1ea6: 'A',
        0x1ea7: 'a',
        0x1ea8: 'A',
        0x1ea9: 'a',
        0x1eaa: 'A',
        0x1eab: 'a',
        0x1eac: 'A',
        0x1ead: 'a',
        0x1eb8: 'E',
        0x1eb9: 'e',
        0x1eba: 'E',
        0x1ebb: 'e',
        0x1ebc: 'E',
        0x1ebd: 'e',
        0x1ebe: 'E',
        0x1ebf: 'e',
        0x1ec0: 'E',
        0x1ec1: 'e',
        0x1ec2: 'E',
        0x1ec3: 'e',
        0x1ec4: 'E',
        0x1ec5: 'e',
        0x1ec6: 'E',
        0x1ec7: 'e',
        0x1ec8: 'I',
        0x1ec9: 'i',
        0x1eca: 'I',
        0x1ecb: 'i',
        0x1ecc: 'O',
        0x1ecd: 'o',
        0x1ece: 'O',
        0x1ecf: 'o',
        0x1ed0: 'O',
        0x1ed1: 'o',
        0x1ed2: 'O',
        0x1ed3: 'o',
        0x1ed4: 'O',
        0x1ed5: 'o',
        0x1ed6: 'O',
        0x1ed7: 'o',
        0x1ed8: 'O',
        0x1ed9: 'o',
        0x1eda: 'O',
        0x1edb: 'o',
        0x1edc: 'O',
        0x1edd: 'o',
        0x1ede: 'O',
        0x1edf: 'o',
        0x1ee0: 'O',
        0x1ee1: 'o',
        0x1ee2: 'O',
        0x1ee3: 'o',
        0x1ee4: 'U',
        0x1ee5: 'u',
        0x1ee6: 'U',
        0x1ee7: 'u',
        0x1ee8: 'U',
        0x1ee9: 'u',
        0x1eea: 'U',
        0x1eeb: 'u',
        0x1eec: 'U',
        0x1eed: 'u',
        0x1eee: 'U',
        0x1eef: 'u',
        0x1ef0: 'U',
        0x1ef1: 'u',
        0x1ef4: 'Y',
        0x1ef5: 'y',
        0x1ef6: 'Y',
        0x1ef7: 'y',
        0x1ef8: 'Y',
        0x1ef9: 'y',
        0x2c60: 'L',
        0x2c61: 'l',
        0x2c62: 'L',
        0x2c63: 'P',
        0x2c64: 'R',
        0x2c65: 'a',
        0x2c66: 't',
        0x2c67: 'H',
        0x2c68: 'h',
        0x2c69: 'K',
        0x2c6a: 'k',
        0x2c6b: 'Z',
        0x2c6c: 'z',
        0xff10: '0',
        0xff11: '1',
        0xff12: '2',
        0xff13: '3',
        0xff14: '4',
        0xff15: '5',
        0xff16: '6',
        0xff17: '7',
        0xff18: '8',
        0xff19: '9',
        0xff21: 'A',
        0xff22: 'B',
        0xff23: 'C',
        0xff24: 'D',
        0xff25: 'E',
        0xff26: 'F',
        0xff27: 'G',
        0xff28: 'H',
        0xff29: 'I',
        0xff2a: 'J',
        0xff2b: 'K',
        0xff2c: 'L',
        0xff2d: 'M',
        0xff2e: 'N',
        0xff2f: 'O',
        0xff30: 'P',
        0xff31: 'Q',
        0xff32: 'R',
        0xff33: 'S',
        0xff34: 'T',
        0xff35: 'U',
        0xff36: 'V',
        0xff37: 'W',
        0xff38: 'X',
        0xff39: 'Y',
        0xff3a: 'Z',
        0xff41: 'a',
        0xff42: 'b',
        0xff43: 'c',
        0xff44: 'd',
        0xff45: 'e',
        0xff46: 'f',
        0xff47: 'g',
        0xff48: 'h',
        0xff49: 'i',
        0xff4a: 'j',
        0xff4b: 'k',
        0xff4c: 'l',
        0xff4d: 'm',
        0xff4e: 'n',
        0xff4f: 'o',
        0xff50: 'p',
        0xff51: 'q',
        0xff52: 'r',
        0xff53: 's',
        0xff54: 't',
        0xff55: 'u',
        0xff56: 'v',
        0xff57: 'w',
        0xff58: 'x',
        0xff59: 'y',
        0xff5a: 'z',

        }

latin_ascii_xlate = defaultdict(lambda: None)

for key, val in list(tmp_latin_ascii_xlate.items()):
    latin_ascii_xlate[key] = str(val)
    latin_ascii_xlate_ext[key] = str(val)

xxlate = {
        0xc0:'A', 0xc1:'A', 0xc2:'A', 0xc3:'A', 0xc4:'A', 0xc5:'A',
        0xc6:'Ae', 0xc7:'C',
        0xc8:'E', 0xc9:'E', 0xca:'E', 0xcb:'E',
        0xcc:'I', 0xcd:'I', 0xce:'I', 0xcf:'I',
        0xd0:'Th', 0xd1:'N',
        0xd2:'O', 0xd3:'O', 0xd4:'O', 0xd5:'O', 0xd6:'O', 0xd8:'O',
        0xd9:'U', 0xda:'U', 0xdb:'U', 0xdc:'U',
        0xdd:'Y', 0xde:'th', 0xdf:'ss',
        0xe0:'a', 0xe1:'a', 0xe2:'a', 0xe3:'a', 0xe4:'a', 0xe5:'a',
        0xe6:'ae', 0xe7:'c',
        0xe8:'e', 0xe9:'e', 0xea:'e', 0xeb:'e',
        0xec:'i', 0xed:'i', 0xee:'i', 0xef:'i',
        0xf0:'th', 0xf1:'n',
        0xf2:'o', 0xf3:'o', 0xf4:'o', 0xf5:'o', 0xf6:'o', 0xf8:'o',
        0xf9:'u', 0xfa:'u', 0xfb:'u', 0xfc:'u',
        0xfd:'y', 0xfe:'th', 0xff:'y',
        0xa1:'!', 0xa2:'{cent}', 0xa3:'{pound}', 0xa4:'{currency}',
        0xa5:'{yen}', 0xa6:'|', 0xa7:'{section}', 0xa8:'{umlaut}',
        0xa9:'{C}', 0xaa:'{^a}', 0xab:'<<', 0xac:'{not}',
        0xad:'-', 0xae:'{R}', 0xaf:'_', 0xb0:'{degrees}',
        0xb1:'{+/-}', 0xb2:'{^2}', 0xb3:'{^3}', 0xb4:"'",
        0xb5:'u', 0xb6:'{paragraph}', 0xb7:' ', 0xb8:'{cedilla}',
        0xb9:'{^1}', 0xba:'{^o}', 0xbb:'>>',
        0xbc:'{1/4}', 0xbd:'{1/2}', 0xbe:'{3/4}', 0xbf:'?',
        0xd7:'x', 0xf7:'/'
        }

for c in ("\N{SPACE}"
          "\N{EXCLAMATION MARK}"
          "\N{QUOTATION MARK}"
          "\N{NUMBER SIGN}"
          "\N{DOLLAR SIGN}"
          "\N{PERCENT SIGN}"
          "\N{AMPERSAND}"
          "\N{APOSTROPHE}"
          "\N{LEFT PARENTHESIS}"
          "\N{RIGHT PARENTHESIS}"
          "\N{ASTERISK}"
          "\N{PLUS SIGN}"
          "\N{COMMA}"
          "\N{HYPHEN-MINUS}"
          "\N{FULL STOP}"
          "\N{SOLIDUS}"
          "\N{DIGIT ZERO}"
          "\N{DIGIT ONE}"
          "\N{DIGIT TWO}"
          "\N{DIGIT THREE}"
          "\N{DIGIT FOUR}"
          "\N{DIGIT FIVE}"
          "\N{DIGIT SIX}"
          "\N{DIGIT SEVEN}"
          "\N{DIGIT EIGHT}"
          "\N{DIGIT NINE}"
          "\N{COLON}"
          "\N{SEMICOLON}"
          "\N{LESS-THAN SIGN}"
          "\N{EQUALS SIGN}"
          "\N{GREATER-THAN SIGN}"
          "\N{QUESTION MARK}"
          "\N{COMMERCIAL AT}"
          "\N{LATIN CAPITAL LETTER A}"
          "\N{LATIN CAPITAL LETTER B}"
          "\N{LATIN CAPITAL LETTER C}"
          "\N{LATIN CAPITAL LETTER D}"
          "\N{LATIN CAPITAL LETTER E}"
          "\N{LATIN CAPITAL LETTER F}"
          "\N{LATIN CAPITAL LETTER G}"
          "\N{LATIN CAPITAL LETTER H}"
          "\N{LATIN CAPITAL LETTER I}"
          "\N{LATIN CAPITAL LETTER J}"
          "\N{LATIN CAPITAL LETTER K}"
          "\N{LATIN CAPITAL LETTER L}"
          "\N{LATIN CAPITAL LETTER M}"
          "\N{LATIN CAPITAL LETTER N}"
          "\N{LATIN CAPITAL LETTER O}"
          "\N{LATIN CAPITAL LETTER P}"
          "\N{LATIN CAPITAL LETTER Q}"
          "\N{LATIN CAPITAL LETTER R}"
          "\N{LATIN CAPITAL LETTER S}"
          "\N{LATIN CAPITAL LETTER T}"
          "\N{LATIN CAPITAL LETTER U}"
          "\N{LATIN CAPITAL LETTER V}"
          "\N{LATIN CAPITAL LETTER W}"
          "\N{LATIN CAPITAL LETTER X}"
          "\N{LATIN CAPITAL LETTER Y}"
          "\N{LATIN CAPITAL LETTER Z}"
          "\N{LEFT SQUARE BRACKET}"
          "\N{REVERSE SOLIDUS}"
          "\N{RIGHT SQUARE BRACKET}"
          "\N{LOW LINE}"
          "\N{LATIN SMALL LETTER A}"
          "\N{LATIN SMALL LETTER B}"
          "\N{LATIN SMALL LETTER C}"
          "\N{LATIN SMALL LETTER D}"
          "\N{LATIN SMALL LETTER E}"
          "\N{LATIN SMALL LETTER F}"
          "\N{LATIN SMALL LETTER G}"
          "\N{LATIN SMALL LETTER H}"
          "\N{LATIN SMALL LETTER I}"
          "\N{LATIN SMALL LETTER J}"
          "\N{LATIN SMALL LETTER K}"
          "\N{LATIN SMALL LETTER L}"
          "\N{LATIN SMALL LETTER M}"
          "\N{LATIN SMALL LETTER N}"
          "\N{LATIN SMALL LETTER O}"
          "\N{LATIN SMALL LETTER P}"
          "\N{LATIN SMALL LETTER Q}"
          "\N{LATIN SMALL LETTER R}"
          "\N{LATIN SMALL LETTER S}"
          "\N{LATIN SMALL LETTER T}"
          "\N{LATIN SMALL LETTER U}"
          "\N{LATIN SMALL LETTER V}"
          "\N{LATIN SMALL LETTER W}"
          "\N{LATIN SMALL LETTER X}"
          "\N{LATIN SMALL LETTER Y}"
          "\N{LATIN SMALL LETTER Z}"
          "\N{VERTICAL LINE}"
          "\N{TILDE}"):
    latin_ascii_xlate_ext[ord(c)] = str(c)
    latin_ascii_xlate[ord(c)] = str(c)

for c in range(0x80):
    if c not in latin_ascii_xlate_ext:
        latin_ascii_xlate_ext[c] = chr(c)

try:
    dummy = set() #Python2.4 >=
except:          #Python2.4 <
    import sets
    set = sets.Set

# Base map
GENERIC_CHAR_MAP = defaultdict(lambda: ' ')
for k, v in list(latin_ascii_xlate_ext.items()):
    GENERIC_CHAR_MAP[k] = v

class LangCharset(object):
    '''
    A langauge Model has a
    1. Set of valid chars of the script : _set or _range ( different representations )
    2. Character transformations to normalize exotic chars : chartranslate
    '''
    def __init__(self, lang, charset=None, charrange=None, chartranslate=None):
        self._set = set()
        self._puncts = frozenset(['!', '"', '%', '&', '(', ')', ',', '-', '.', '/', ':', '?', '\\'])
        self._puncts_backup = {}
        self._range = []
        self.lang = lang
        self._final_dict = defaultdict(lambda: ' ')
        if charset:
            self._set = charset
            self.is_in = self.is_in_set
        if chartranslate:
            self._translate = chartranslate
        if charrange:
            self._range = charrange
            self.is_in = self.is_in_range

        # Initialize the translate string only for eng while init
        # Deal with the other language on first call to lang_clean
        if lang == "eng":
            self.activate_final_transform_dict()

    def initialize_range_lang_translate_dict(self):
        for start, end in self._range:
            start, end = ord(start), ord(end)
            for char in range(start, end + 1):
                self._final_dict[char] = chr(char)

    def activate_final_transform_dict(self):
        '''
        Load the translate dict to only keep the charactes in the lang charser.
        Loads dict with sys.maxunicode keys, choosing speed over memory.
        '''
        if self._set:
            self._final_dict = defaultdict(lambda: ' ')
            for each in self._set:
                self._final_dict[ord(each)] = each
        else:
            self.initialize_range_lang_translate_dict()

    def update_final_dict_with_punct(self, restore):
        for each in self._puncts:
            k = ord(each)
            if restore:
                self._final_dict[k] = self._puncts_backup[k]
            else:
                self._puncts_backup[k] = self._final_dict[k]
                self._final_dict[k] = each

    def is_in_set(self, char):
        return char in self._set

    def is_in_range(self, char):
        for start, end in self._range:
            if start <= char <= end:
                return True
        return False

    def char_map(self, char):
        if ord(char) in self._translate:
            return char.translate(self._translate)
        else:
            return ' '

    def lang_clean(self, in_string, make_lower, allow_punct, string_transform_func=None, only_norm=False):
        '''
        '''
        # Load self._final translate dict only on first invocation.
        # Except "eng", which we load on init
        if not self._final_dict:
            self.activate_final_transform_dict()

        # Step 1.
        # Normalize exotic chars to their ascii variants mostly.
        out_string = in_string.translate(self._translate)

        # Step 2.
        if make_lower:
            out_string = out_string.lower()

        # Step 3.
        # Apply any langauge specific multichar string to string transforms
        if string_transform_func:
            out_string = string_transform_func(out_string)

        # Step 4.
        # Cull characters not in the lang charset.
        if only_norm:
            out_string = ''.join( [x if x.isalnum() else ' '  for x in out_string])
        else:
            if not allow_punct:
                out_string = out_string.translate(self._final_dict)
            else:
                self.update_final_dict_with_punct(restore=False)
                out_string = out_string.translate(self._final_dict)
                self.update_final_dict_with_punct(restore=True)
        return out_string

def construct_char_map_set(char_str, base_char_set = set()):
    '''
    A complete set of charactets used in the language script.
    This always includes English Alphabets. Yes, we are biased.
    '''
    char_list = [ x.strip() for x in char_str.split(',') ]
    char_list_as_string = ''.join(char_list)
    new_char_list = set(char_list + list(char_list_as_string.upper()) + list(char_list_as_string.lower()))
    return new_char_list | base_char_set

def construct_translate_map(lang_translate_map=[], base_dict = dict(), charset = set()):
    '''
    Has a complete translation dict for the langauge.
    '''
    map_dict = {}
    for k, v in list(base_dict.items()):
        map_dict[k] = v
    for k, v in lang_translate_map:
        map_dict[ord(k)] = v
    for each in charset:
        map_dict[ord(each)] = each
    return map_dict

ENG_CHAR_MAP_SET         = set(map(str, string.digits + string.ascii_letters))
ENG_TRANSLATE_CHAR_MAP   = construct_translate_map([("Ÿ", "Y"), ("Œ", "OE"), ("ÿ", "y"), ("œ", "oe")], base_dict=GENERIC_CHAR_MAP)

FRA_CHAR_MAP_SET         = construct_char_map_set("à,â,ç,é,è,ê,ë,î,ï,ô,ù,û,ü,ÿ,æ,œ", ENG_CHAR_MAP_SET)
FRA_TRANSLATE_CHAR_MAP   = construct_translate_map(lang_translate_map=[("’", ""), ("'", "")], base_dict=ENG_TRANSLATE_CHAR_MAP, charset=FRA_CHAR_MAP_SET)

DEU_CHAR_MAP_SET         = construct_char_map_set("ä,ö,ü,ß", ENG_CHAR_MAP_SET)
DEU_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=DEU_CHAR_MAP_SET)

NOR_CHAR_MAP_SET         = construct_char_map_set("æ,å,ø", ENG_CHAR_MAP_SET)
NOR_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=NOR_CHAR_MAP_SET)

SPA_CHAR_MAP_SET         = construct_char_map_set("á,é,í,ï,ó,ú,ü,ñ", ENG_CHAR_MAP_SET)
SPA_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=SPA_CHAR_MAP_SET)

NLD_CHAR_MAP_SET         = construct_char_map_set("ĳ", ENG_CHAR_MAP_SET)
NLD_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=NLD_CHAR_MAP_SET)

DAN_CHAR_MAP_SET         = construct_char_map_set("æ,å,ø", ENG_CHAR_MAP_SET)
DAN_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=DAN_CHAR_MAP_SET)

SWE_CHAR_MAP_SET         = construct_char_map_set("á,à,ä,å,é,è,ö,ü", ENG_CHAR_MAP_SET)
SWE_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=SWE_CHAR_MAP_SET)

FIN_CHAR_MAP_SET         = construct_char_map_set("ä,å,ö,š,ž", ENG_CHAR_MAP_SET)
FIN_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=FIN_CHAR_MAP_SET)

ITA_CHAR_MAP_SET         = construct_char_map_set("à,é,è,ì,î,ó,ò,ù", ENG_CHAR_MAP_SET)
ITA_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=ITA_CHAR_MAP_SET)

POR_CHAR_MAP_SET         = construct_char_map_set("ã,õ,á,é,í,ó,ú,â,ê,ô,à,ç", ENG_CHAR_MAP_SET)
POR_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=POR_CHAR_MAP_SET)

RUS_CHAR_MAP_SET         = construct_char_map_set("а,б,в,г,д,е,ё,ж,з,и,й,к,л,м,н,о,п,р,с,т,у,ф,х,ц,ч,ш,щ,ъ,ы,ь,э,ю,я", ENG_CHAR_MAP_SET)
RUS_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=RUS_CHAR_MAP_SET)

POL_CHAR_MAP_SET         = construct_char_map_set("ą,ć,ę,ł,ń,ó,ś,ź,ż", ENG_CHAR_MAP_SET)
POL_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=POL_CHAR_MAP_SET)

TUR_CHAR_MAP_SET         = construct_char_map_set("ç,ğ,ı,ö,ş,ü", ENG_CHAR_MAP_SET)
TUR_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=TUR_CHAR_MAP_SET)

UKR_CHAR_MAP_SET         = construct_char_map_set("б,г,ґ,д,ж,з,к,л,м,н,п,р,с,т,ф,х,ц,ч,ш,щ,а,е,є,и,і,ї,о,у,ю,я,й,в", ENG_CHAR_MAP_SET)
UKR_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=UKR_CHAR_MAP_SET)

CAT_CHAR_MAP_SET         = construct_char_map_set("à,é,è,í,ï,ó,ò,ú,ü,ç", ENG_CHAR_MAP_SET)
CAT_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=CAT_CHAR_MAP_SET)

VIE_CHAR_MAP_SET         = construct_char_map_set("ă,ỹ,ử,đ,ũ,ạ,ỡ,ả,ấ,ầ,ẩ,ẫ,ậ,ỳ,ắ,ằ,ư,ẳ,ẵ,ặ,ẹ,ỵ,ẻ,ẽ,ế,ề,ơ,ể,ễ,ệ,ỉ,ị,ọ,ợ,ỏ,ố,ứ,ồ,ổ,ỗ,ộ,ớ,ờ,ở,á,à,ã,â,ụ,ủ,é,è,ừ,ê,í,ì,ữ,ự,ĩ,ó,ò,õ,ô,ỷ,ù,ú,ý", ENG_CHAR_MAP_SET)
VIE_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=VIE_CHAR_MAP_SET)

IND_CHAR_MAP_SET         = ENG_CHAR_MAP_SET
IND_TRANSLATE_CHAR_MAP   = ENG_TRANSLATE_CHAR_MAP

RON_CHAR_MAP_SET         = construct_char_map_set("ă,â,î,ș,ț", ENG_CHAR_MAP_SET)
RON_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=RON_CHAR_MAP_SET)

HUN_CHAR_MAP_SET         = construct_char_map_set("á,é,í,ó,ö,ő,ú,ü,ű", ENG_CHAR_MAP_SET)
HUN_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=HUN_CHAR_MAP_SET)

CES_CHAR_MAP_SET         = construct_char_map_set("á,č,ď,é,ě,í,ň,ó,ř,š,ť,ú,ů,ý,ž", ENG_CHAR_MAP_SET)
CES_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=CES_CHAR_MAP_SET)

HBS_CHAR_MAP_SET         = construct_char_map_set("ǆ,ǉ,ǌ,ć,č,đ,š,ž,а,б,в,г,д,е,ж,з,и,к,л,м,н,о,п,р,с,т,у,ф,х,ц,ч,ш,ђ,ј,љ,њ,ћ,џ", ENG_CHAR_MAP_SET)
HBS_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=HBS_CHAR_MAP_SET)

SRP_CHAR_MAP_SET         = construct_char_map_set("ǆ,ǉ,ǌ,ć,č,đ,š,ž,а,б,в,г,д,е,ж,з,и,к,л,м,н,о,п,р,с,т,у,ф,х,ц,ч,ш,ђ,ј,љ,њ,ћ,џ", ENG_CHAR_MAP_SET)
SRP_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=SRP_CHAR_MAP_SET)

HRV_CHAR_MAP_SET         = construct_char_map_set("ǆ,ǉ,ǌ,ć,č,đ,š,ž", ENG_CHAR_MAP_SET)
HRV_TRANSLATE_CHAR_MAP   = construct_translate_map(base_dict=ENG_TRANSLATE_CHAR_MAP, charset=HRV_CHAR_MAP_SET)

ZHO_CHAR_RANGE    = [('a', 'z'), ('A', 'Z'), ('0', '9'), ('\u4E00', '\u9FFF'), ('\u3400', '\u4DFF'), ('\U00020000', '\U0002A6DF')]
_ZHO_TRANSLATE_CHAR_MAP    = [("．", " "), ("·", " "), ("—", " "), ("(", " "), (")", " "), ("[", " "), ("]", " "), \
                               ("《", " "), ("》", " "), ("・", " "), ("（", " "), ("）", " "), ("「", " "), ("」", " "),\
                               ("『", " "), ("』", " "), ("（", " "), ("）", " "), ("〔", " "), ("〕", " "), ("［", " "),\
                               ("］", " "), ("｛", " "), ("｝", " "), ("〈", " "), ("〉", " "), ("《", " "), ("》", " "),\
                               ("【", " "), ("】", " "), ("〖", " "), ("〗", " "), ("〘", " "), ("〙", " "), ("〚", " "),\
                               ("〛", " "), ("゠", " "), ("＝", " "), ("　", " "), ("?", " "), ("、", " "), ("。", " "),\
                              ("“", " "), ("”", " "), ("！", " "), ("，", " "), ("：", " "), ("；", " "), ("？", " "),]
ZHO_TRANSLATE_CHAR_MAP = construct_translate_map(_ZHO_TRANSLATE_CHAR_MAP, ENG_TRANSLATE_CHAR_MAP)

# Japanese
# Hiragana ( 3040 - 309f)
# Katakana ( 30a0 - 30ff)
# CJK unifed ideographs - Common and uncommon kanji ( 4e00 - 9faf)
JPN_CHAR_RANGE    = [('a', 'z'), ('A', 'Z'), ('0', '9'), ('\u3040', '\u309f'), ('\u30a0', '\u30ff'), ('\u4E00', '\u9FFF'),]
_JPN_TRANSLATE_CHAR_MAP    = [("．", " "), ("·", " "), ("—", " "), ("(", " "), (")", " "), ("[", " "), ("]", " "), \
                               ("《", " "), ("》", " "), ("・", " "), ("（", " "), ("）", " "), ("「", " "), ("」", " "),\
                               ("『", " "), ("』", " "), ("（", " "), ("）", " "), ("〔", " "), ("〕", " "), ("［", " "),\
                               ("］", " "), ("｛", " "), ("｝", " "), ("〈", " "), ("〉", " "), ("《", " "), ("》", " "),\
                               ("【", " "), ("】", " "), ("〖", " "), ("〗", " "), ("〘", " "), ("〙", " "), ("〚", " "),\
                               ("〛", " "), ("゠", " "), ("＝", " "), ("　", " "), ("?", " "), ("、", " "), ("。", " "),\
                              ("“", " "), ("”", " "), ("！", " "), ("，", " "), ("：", " "), ("；", " "), ("？", " "),]

JPN_TRANSLATE_CHAR_MAP = construct_translate_map(_JPN_TRANSLATE_CHAR_MAP, ENG_TRANSLATE_CHAR_MAP)

# Korean Hangul
# Hangul Syllables (AC00-D7A3)
# Hangul Jamo (1100–11FF)
# Hangul Compatibility Jamo (3130-318F)
# Hangul Jamo Extended-A (A960-A97F)
# Hangul Jamo Extended-B (D7B0-D7FF)
KOR_CHAR_RANGE    = [('a', 'z'), ('A', 'Z'), ('0', '9'), ('\uac00', '\ud7a3'), ('\u1100', '\u11ff'), ('\u3130', '\u318f'), ('\ua960', '\ua97f'), ('\ud7b0', '\ud7ff'),]
_KOR_TRANSLATE_CHAR_MAP    = [("．", " "), ("·", " "), ("—", " "), ("(", " "), (")", " "), ("[", " "), ("]", " "), \
                               ("《", " "), ("》", " "), ("・", " "), ("（", " "), ("）", " "), ("「", " "), ("」", " "),\
                               ("『", " "), ("』", " "), ("（", " "), ("）", " "), ("〔", " "), ("〕", " "), ("［", " "),\
                               ("］", " "), ("｛", " "), ("｝", " "), ("〈", " "), ("〉", " "), ("《", " "), ("》", " "),\
                               ("【", " "), ("】", " "), ("〖", " "), ("〗", " "), ("〘", " "), ("〙", " "), ("〚", " "),\
                               ("〛", " "), ("゠", " "), ("＝", " "), ("　", " "), ("?", " "), ("、", " "), ("。", " "),\
                              ("“", " "), ("”", " "), ("！", " "), ("，", " "), ("：", " "), ("；", " "), ("？", " "),]

KOR_TRANSLATE_CHAR_MAP = construct_translate_map(_KOR_TRANSLATE_CHAR_MAP, ENG_TRANSLATE_CHAR_MAP)

CEB_CHAR_MAP_SET         = ENG_CHAR_MAP_SET
CEB_TRANSLATE_CHAR_MAP   = ENG_TRANSLATE_CHAR_MAP

LANG_ISO_TO_NAME_DICT = dict(
        [('eng', 'english'), ('fra', 'french'), ('deu', 'german'), ('nor', 'norwegian'), ('spa', 'spanish'),
         ('nld', 'dutch'), ('dan', 'danish'), ('swe', 'swedish'), ('fin', 'finnish'), ('ita', 'italian'),
         ('por', 'portuguese'), ('rus', 'russian'),  ('pol', 'polish'), ('tur', 'turkish'), ('zho', 'chinese'),
         ('jpn', 'japanese'), ('kor', 'korean'), ('ces', 'czech'), ('ukr', 'ukrainian'), ('cat', 'catalan'),
         ('vie', 'vietnamese'), ('ind', 'indonesian'), ('ron', 'romanian'), ('hun', 'hungarian'),
         ('srp', 'serbian'), ('hrv', 'croatian')])

LANG_NAME_TO_ISO_DICT = dict((v, k) for k, v in list(LANG_ISO_TO_NAME_DICT.items()))

# Language mappings
LANG_MAP = {
        'eng': LangCharset('eng', charset=ENG_CHAR_MAP_SET, chartranslate=ENG_TRANSLATE_CHAR_MAP),
        'fra': LangCharset('fra', charset=FRA_CHAR_MAP_SET, chartranslate=FRA_TRANSLATE_CHAR_MAP),
        'deu': LangCharset('deu', charset=DEU_CHAR_MAP_SET, chartranslate=DEU_TRANSLATE_CHAR_MAP),
        'nor': LangCharset('nor', charset=NOR_CHAR_MAP_SET, chartranslate=NOR_TRANSLATE_CHAR_MAP),
        'spa': LangCharset('spa', charset=SPA_CHAR_MAP_SET, chartranslate=SPA_TRANSLATE_CHAR_MAP),
        'nld': LangCharset('nld', charset=NLD_CHAR_MAP_SET, chartranslate=NLD_TRANSLATE_CHAR_MAP),
        'dan': LangCharset('dan', charset=DAN_CHAR_MAP_SET, chartranslate=DAN_TRANSLATE_CHAR_MAP),
        'swe': LangCharset('swe', charset=SWE_CHAR_MAP_SET, chartranslate=SWE_TRANSLATE_CHAR_MAP),
        'fin': LangCharset('fin', charset=FIN_CHAR_MAP_SET, chartranslate=FIN_TRANSLATE_CHAR_MAP),
        'ita': LangCharset('ita', charset=ITA_CHAR_MAP_SET, chartranslate=ITA_TRANSLATE_CHAR_MAP),
        'por': LangCharset('por', charset=POR_CHAR_MAP_SET, chartranslate=POR_TRANSLATE_CHAR_MAP),
        'rus': LangCharset('rus', charset=RUS_CHAR_MAP_SET, chartranslate=RUS_TRANSLATE_CHAR_MAP),
        'pol': LangCharset('pol', charset=POL_CHAR_MAP_SET, chartranslate=POL_TRANSLATE_CHAR_MAP),
        'tur': LangCharset('tur', charset=TUR_CHAR_MAP_SET, chartranslate=TUR_TRANSLATE_CHAR_MAP),
        'zho': LangCharset('zho', charrange=ZHO_CHAR_RANGE, chartranslate=ZHO_TRANSLATE_CHAR_MAP),
        'jpn': LangCharset('jpn', charrange=JPN_CHAR_RANGE, chartranslate=JPN_TRANSLATE_CHAR_MAP),
        'kor': LangCharset('kor', charrange=KOR_CHAR_RANGE, chartranslate=KOR_TRANSLATE_CHAR_MAP),
        'ceb': LangCharset('ceb', charset=CEB_CHAR_MAP_SET, chartranslate=CEB_TRANSLATE_CHAR_MAP),
        'ukr': LangCharset('ukr', charset=UKR_CHAR_MAP_SET, chartranslate=UKR_TRANSLATE_CHAR_MAP),
        'cat': LangCharset('ukr', charset=CAT_CHAR_MAP_SET, chartranslate=CAT_TRANSLATE_CHAR_MAP),
        'vie': LangCharset('ukr', charset=VIE_CHAR_MAP_SET, chartranslate=VIE_TRANSLATE_CHAR_MAP),
        'ind': LangCharset('ind', charset=IND_CHAR_MAP_SET, chartranslate=IND_TRANSLATE_CHAR_MAP),
        'ron': LangCharset('ron', charset=RON_CHAR_MAP_SET, chartranslate=RON_TRANSLATE_CHAR_MAP),
        'hun': LangCharset('hun', charset=HUN_CHAR_MAP_SET, chartranslate=HUN_TRANSLATE_CHAR_MAP),
        'ces': LangCharset('ces', charset=CES_CHAR_MAP_SET, chartranslate=CES_TRANSLATE_CHAR_MAP),
        'hbs': LangCharset('hbs', charset=HBS_CHAR_MAP_SET, chartranslate=HBS_TRANSLATE_CHAR_MAP),
        'srp': LangCharset('srp', charset=SRP_CHAR_MAP_SET, chartranslate=SRP_TRANSLATE_CHAR_MAP),
        'hrv': LangCharset('hrv', charset=HRV_CHAR_MAP_SET, chartranslate=HRV_TRANSLATE_CHAR_MAP),
        }

# For wiki datagen : which still uses old codes.
for old_code, new_code in [('dut', 'nld'),
                           ('ger', 'deu')]:
    LANG_MAP[old_code] = LANG_MAP[new_code]

# Language list
LANG_LIST = ['cat', 'ces', 'dan', 'nld', 'eng', 'fin', 'fra', 'deu', 'hun', 'hrv',
             'ind', 'ita', 'jpn', 'kor', 'nor', 'pol', 'por', 'ron', 'rus', 'spa',
             'srp', 'swe', 'tur', 'ukr', 'vie', 'zho']

def print_char_map_set(char_map_set):
    if not char_map_set:
        print(None)
    for x in sorted(char_map_set):
        if x not in GENERIC_CHAR_MAP:
            print(x)

def print_char_map_range(char_map_range):
    if not char_map_range:
        print(None)
    for x in sorted(char_map_range):
        print(("From %s to %s" % x))

def print_translate_map(char_map):
    for x, y in list(char_map.items()):
        if x not in GENERIC_CHAR_MAP:
            x = chr(x)
            print((x, '->', '\'', y, '\''))

def escaper(char):
    if char == '\\':
        return '\\\\'
    elif char == '"':
        return '\\"'
    elif char == '\'':
        return '\\\''
    elif char == '$':
        return '\$'
    return char

def generate_perl_map(lang_map):
    languages = sorted(LANG_ISO_TO_NAME_DICT.keys())

    print(perl_file_header)
    print('package LangUtils;\n')
    print('use utf8;')
    print('use warnings;')
    print('use strict;')
    print(perl_translate_string)
    print(perl_functions)
    for name in languages:
        lang_obj = lang_map[name]
        full_name = LANG_ISO_TO_NAME_DICT.get(name)
        new_all_chars_set = None
        new_all_chars_range = None

        print(('# %s chars sets' % full_name.capitalize()))
        if lang_obj._set:
            new_all_chars_set =  ',\t'.join(["'%s', 1" % escaper(k) for k in lang_obj._set])
            print(('my %%%s_char_set = (%s);' % (name, new_all_chars_set.encode('utf8'))))
        if lang_obj._range:
            new_all_chars_range = ', '.join([ "['%s', '%s']" % each_range for each_range in lang_obj._range])
            print(('my @%s_char_range = (%s);' % (name, new_all_chars_range.encode('utf8'))))
        new_all_mapping =  ',\t'.join(['"%s","%s"' % (escaper(chr(k)), escaper(v)) for k, v in sorted(iter(list(lang_obj._translate.items())), key=itemgetter(1))])
        print(('my %%%s_mapping = (%s);' % (name, new_all_mapping.encode('utf8'))))

        if lang_obj._range:
            print(('my $%s_language_data = create_char_map_range_obj(\\@%s_char_range, \\%%%s_mapping);' % tuple([name] * 3)))
        if lang_obj._set:
            print(('my $%s_language_data = create_char_map_set_obj(\\%%%s_char_set, \\%%%s_mapping);' % tuple([name] * 3)))

    print('''
# Special case to handle cleanString2 with '.'
$eng_char_set{ord('.')} = 1;
    ''')
    print('# Language char mapping')
    print('our %lang_clean_maps = (')
    for name in languages:
        print(("    '%s' => $%s_language_data," % (name, name)))
    print(');')
    print('''
sub get_lang_map { return %lang_clean_maps;}
sub get_translate_string_map { return %translate_string;}
1;''')

def read_json_file(filePath):
    fp = open(filePath, 'r')
    data = fp.read()
    fp.close()
    return eval(data)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'genperl':
        generate_perl_map(LANG_MAP)
    else:
        for name, v in list(LANG_MAP.items()):
            name = name.upper()
            print(('Charset : ', name))
            print_char_map_set(v._set)
            print(('Char range: ', name))
            print_char_map_range(v._range)
            print(('TRANSLATE MAP: ', name))
            print_translate_map(v._translate)
