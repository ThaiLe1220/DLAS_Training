""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
from unidecode import unidecode
from underthesea import word_tokenize, text_normalize

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("tp", "thành phố"),
        ("hcm", "Hồ Chí Minh"),
        ("ubnd", "Ủy ban Nhân dân"),
        ("qd", "quyết định"),
        ("kt", "kiểm tra"),
        ("xh", "xã hội"),
        ("nv", "nhân viên"),
        ("bv", "bệnh viện"),
        ("hs", "học sinh"),
        ("gv", "giáo viên"),
        ("pgd", "phòng giáo dục"),
        ("cv", "công việc"),
        ("ld", "lao động"),
        ("dn", "doanh nghiệp"),
        ("tt", "trung tâm"),
        ("cn", "công nhân"),
        ("qg", "quốc gia"),
        ("dt", "dân tộc"),
        ("qdnd", "quân đội nhân dân"),
        ("cs", "công an"),
        ("khcn", "khoa học công nghệ"),
        ("ct", "chương trình"),
        ("ht", "hệ thống"),
        ("kd", "kinh doanh"),
        ("qt", "quốc tế"),
        ("dl", "du lịch"),
        ("tttm", "trung tâm thương mại"),
        ("tt", "thông tin"),
        ("dt", "điện thoại"),
        ("clb", "câu lạc bộ"),
        ("ktxh", "kinh tế xã hội"),
        ("qdkt", "quyết định kiểm tra"),
        ("ndtb", "nông dân tiêu biểu"),
        ("ubql", "Ủy ban quản lý"),
        ("bqlkcn", "Ban quản lý khu công nghiệp"),
        ("ldxh", "lao động xã hội"),
        ("khxh", "khoa học xã hội"),
        ("bvdl", "Bệnh viện đa khoa"),
        ("ttyt", "Trung tâm y tế"),
        ("svdh", "sinh viên đại học"),
        ("kts", "kiến trúc sư"),
        ("tgd", "Tổng giám đốc"),
        ("tmdt", "thương mại điện tử"),
        ("ttxvn", "Thông tấn xã Việt Nam"),
        ("tttt", "thông tin truyền thông"),
        ("qhqt", "quan hệ quốc tế"),
        ("tdtt", "thể dục thể thao"),
        ("httt", "hệ thống thông tin"),
        ("qhgq", "quy hoạch giao thông"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def normalize_numbers(text):
    number_map = {
        "0": "không",
        "1": "một",
        "2": "hai",
        "3": "ba",
        "4": "bốn",
        "5": "năm",
        "6": "sáu",
        "7": "bảy",
        "8": "tám",
        "9": "chín",
    }
    for digit, word in number_map.items():
        text = re.sub(r"\b" + digit + r"\b", word, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def normalize_text(text):
    """Normalize text using underthesea's text_normalize."""
    return text_normalize(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = normalize_text(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = normalize_text(text)
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for Vietnamese text, including number and abbreviation expansion."""
    text = normalize_text(text)
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text
