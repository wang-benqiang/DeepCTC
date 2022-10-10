import json
import re
from copy import deepcopy
from pypinyin import pinyin, lazy_pinyin, Style
from utils.langconv import *

PUNC_RE = re.compile(
    r"[,.，–\'。≧▽ω≦з」∠＝｜|／°^＜＞〉{}．😊😄ㄛ丨⋯％😭😳😠；*＂=－『』😍「→～—￥＋×∩╭╮`~$%…&:：!！【】@‘’“”\-_\\/、\]\[#+~·《》()（）]+"
)
ENG_DIGIT_RE = re.compile(r'[a-zA-Z]|[0-9]')
CN_DIGIT_RE = re.compile(r'[一二三四五六七八九十]')
CN_RE = re.compile(r'[\u4e00-\u9fa5]')

# 带空格' '
SPACE_SIGNS = (' ', '\xa0', '\u3000', '\ufeff', '\xc2', '\t', '\n', '\u202A',
               '\u202a', '\u202c', '\r', '\u200b', '\u200d', '\ud83d', '\udc49',
               '\ud83e', '\udd29', '\ud83c', '\udf32', '\ud83d', '\udc47',
               '\udc0e', '\udf5c', '\udc26', '\udcaa', '\udf1f', '\udc4d', '\udca4',
               '\ude18', '\udf19', '\udd38', '\udf89', '\x08', '\ufe0f', '\ue248',
               '\u3000','\u2002', )
# 不带空格，在模型中赋予空格语义
SPACE_SIGNS_2 = ('\xa0', '\u3000', '\ufeff', '\xc2', '\t', '\n', '\u202A',
                 '\u202a', '\u202c', '\r', '\u200b', '\u200d', '\ud83d', '\udc49',
                 '\ud83e', '\udd29', '\ud83c', '\udf32', '\ud83d', '\udc47',
                 '\udc0e', '\udf5c', '\udc26', '\udcaa', '\udf1f', '\udc4d', '\udca4',
                 '\ude18', '\udf19', '\udd38', '\udf89', '\x08', '\ufe0f', '\ue248',
                 '\u3000', '\u2002', )
QUANT_CN_SIGN = json.load(
    open('src/utils/pseudo_data/data/quantifier_char.json', 'r', encoding='utf8'))
SPACE_RE = re.compile('[{}]'.format('|'.join(SPACE_SIGNS)))
SPACE_RE_2 = re.compile('[{}]'.format('|'.join(SPACE_SIGNS_2)))
QUANT_CN_RE = re.compile('[{}]'.format('|'.join(QUANT_CN_SIGN)))

QUOTATION_CONTENT_RE = re.compile(
    r"[‘'\“\"\(\《\<](.{1,8}?)[’'\”\"\)\》\>]")  # 中英文单双引号/括号/书名号/   内容 内容长度:1-6


ORDER_RE1 = re.compile(r'^[\d一二三四五六七八九十][、,,. )）。]?')
ORDER_RE2 = re.compile(r'^[(（[【][、,,. 。\d一二三四五六七八九十]?[)）\]】、,,. 。]?')


def remove_order(text):
    "移除1、 1, 之类的序号, 序号数字只能为个数位"
    text2 = re.sub(ORDER_RE2, '', text)
    if text2 != text:
        return text2
    text = re.sub(ORDER_RE1, '', text)
    return text


def remove_en_digit(text):
    text = re.sub(ENG_DIGIT_RE, '', text)
    return text


def replace_char(old_string, char, index):
    '''
    字符串按索引位置替换字符
    '''
    if isinstance(old_string, list):
        old_string = str(old_string)
    # 新的字符串 = 老字符串[:要替换的索引位置] + 替换成的目标字符 + 老字符串[要替换的索引位置+1:]
    if isinstance(index, int):
        new_string = old_string[:index] + char + old_string[index+1:]
    else:
        # index是一个range的情况
        index_range_len = index[1]-index[0]
        if len(char) == 1:
            char = char*index_range_len
        new_string = old_string[:index[0]] + char + \
            old_string[index[0]+index_range_len:]
    return new_string


def replace_list_ele(li, eles, index_range):
    """_summary_

    Args:
        li (_type_): [1,2,3,4]
        eles (_type_): [8,9]
        index_range (_type_): [1,3]

    Returns:
        _type_: [1,8,9,4]
    """
    li = deepcopy(li)
    length = index_range[-1] - index_range[0]
    if len(eles) <= 1:
        eles = [eles] * length
    for i, idx in enumerate(range(index_range[0], index_range[1])):
        li[idx] = eles[i]
    return li


def remove_space(text):
    text = re.sub(SPACE_RE, '', text)
    return text


def remove_space_2(text):
    text = re.sub(SPACE_RE_2, '', text)
    return text


def replace_punc_for_bert(text):
    text = text.replace('…', '.').replace("‘", "'").replace("’", "'").replace(
        '“', '"').replace('”', '"').replace('|', '｜')
    text = remove_space(text).lower().replace(
        '[mask]', '[MASK]')  # take care of [MASK]
    return text


def replace_punc_for_bert_keep_space(text):
    text = text.replace('…', '.').replace("‘", "'").replace("’", "'").replace(
        '“', '"').replace('”', '"').replace('|', '｜')
    text = remove_space_2(text).lower().replace(
        '[mask]', '[MASK]')  # take care of [MASK]
    return text


def tradition_to_simple(text: str):
    """繁体转简体"""
    text = Converter('zh-hans').convert(text)
    return text


def inclue_punc(text):
    if len(PUNC_RE.findall(text)) > 0:
        return True
    return False


def inclue_cn_digit(text):
    if len(CN_DIGIT_RE.findall(text)) > 0:
        return True
    return False


def include_eng_digit_char(text):
    if len(ENG_DIGIT_RE.findall(text)) > 0:
        return True
    return False


def include_cn(text):
    if len(CN_RE.findall(text)) > 0:
        return True
    return False


def include_quant_cn(text):
    if len(QUANT_CN_RE.findall(text)) > 0:
        return True
    return False


def change_ele_order_in_list(src, trg,
                             a_start_idx, b_start_idx,
                             size,
                             max_len):

    c = src[a_start_idx+size:max_len] + trg[b_start_idx:b_start_idx+size]
    return c

def get_cn_pinyin_first_letter(text):
    pys = [i[0] for i in pinyin(text, style=Style.FIRST_LETTER)]
    return ''.join(pys)


if __name__ == '__main__':
    # r = ' '.join(replace_punc_for_bert('[MASK]你好a')).replace(
    #     '[ M A S K ]', '[MASK]')
    r = get_cn_pinyin_first_letter('三哦东西')
    print(r)
