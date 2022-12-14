import argparse
import re
from collections import defaultdict


def read_cilin(filename):
    word_to_class = defaultdict(list)
    class_to_word = {}
    with open(filename, "r", encoding="gbk") as f:
        for line in f:
            li = line.strip().split()
            class_to_word[li[0]] = li[1:]
            for word in li[1:]:
                word_to_class[word].append(li[0])
    return word_to_class, class_to_word


def extend_pattern(filename):
    word_to_class, class_to_word = read_cilin("cilin.txt")
    with open(filename, "r") as f:
        chunks = f.read().split("\n\n")
        for chunk in chunks:
            span1, span2, correct = chunk.split("\n")
            if span1 in word_to_class.keys() and len(word_to_class[span1]) == 1:
                print(" ".join(class_to_word[word_to_class[span1][0]]))
            if span2 in word_to_class.keys() and len(word_to_class[span2]) == 1:
                print(" ".join(class_to_word[word_to_class[span2][0]]))


def load_pattern(filename):
    pattern_set = set()
    with open(filename, "r") as f:
        chunks = f.read().split("\n\n")
        for chunk in chunks:
            span1, span2, correct_way = chunk.split("\n")
            regular_expression = span1 + ".*" + span2
            pattern_set.add((re.compile(regular_expression),
                            span1, span2, correct_way))
            if correct_way == "3":
                regular_expression = span2 + ".*" + span1
                pattern_set.add(
                    (re.compile(regular_expression), span2, span1, correct_way))
    return pattern_set


filter_token = ["。", "，", "：", "、", "？", "！", "及", "和", "或", "而", "且", "但"]


def correct(input_file, output_file, pattern_file):
    sentence_num, error_num = 0, 0
    pattern_set = list(load_pattern(pattern_file))
    pattern_set.sort(key=lambda x: len(x[1]) + len(x[2]), reverse=True)
    with open(input_file, "r") as f:
        with open(output_file, "w") as o:
            for line in f:
                line = line.strip()
                is_corrected = False
                for pattern in pattern_set:
                    r_e = pattern[0]
                    res = r_e.findall(line)
                    for error_span in res:
                        b = False
                        for t in filter_token:
                            if t in error_span:
                                b = True
                                break
                        if not b:
                            correct_way = pattern[-1]
                            if correct_way == "0":
                                continue
                            elif correct_way == "1":
                                correct_span = error_span.lstrip(pattern[1])
                            elif correct_way == "2":
                                correct_span = error_span.rstrip(pattern[2])
                            elif correct_way == "3":
                                if len(pattern[1]) < len(pattern[2]):
                                    correct_span = error_span.lstrip(
                                        pattern[1])
                                else:
                                    correct_span = error_span.rstrip(
                                        pattern[2])
                            else:
                                c, t = correct_way.split()
                                if c == "1":
                                    correct_span = t + \
                                        error_span.lstrip(pattern[1])
                                else:
                                    correct_span = error_span.rstrip(
                                        pattern[2]) + t
                            is_corrected = True
                            error_num += 1
                            line = line.replace(error_span, correct_span)
                if is_corrected:
                    sentence_num += 1
                o.write(line + '\n')
    return sentence_num, error_num


def gec_pattern_correct(src_texts, pattern_file='src/vocab/error_templates_500.txt'):
    sentence_num, error_num = 0, 0
    pattern_set = list(load_pattern(pattern_file))
    pattern_set.sort(key=lambda x: len(x[1]) + len(x[2]), reverse=True)
    pred_texts = []
    if isinstance(src_texts, str):
      src_texts = [src_texts]
    for line in src_texts:
        line = line.strip()
        is_corrected = False
        for pattern in pattern_set:
            r_e = pattern[0]
            res = r_e.findall(line)
            for error_span in res:
                b = False
                for t in filter_token:
                    if t in error_span:
                        b = True
                        break
                if not b:
                    correct_way = pattern[-1]
                    if correct_way == "0":
                        continue
                    elif correct_way == "1":
                        correct_span = error_span.lstrip(pattern[1])
                    elif correct_way == "2":
                        correct_span = error_span.rstrip(pattern[2])
                    elif correct_way == "3":
                        if len(pattern[1]) < len(pattern[2]):
                            correct_span = error_span.lstrip(
                                pattern[1])
                        else:
                            correct_span = error_span.rstrip(
                                pattern[2])
                    else:
                        c, t = correct_way.split()
                        if c == "1":
                            correct_span = t + \
                                error_span.lstrip(pattern[1])
                        else:
                            correct_span = error_span.rstrip(
                                pattern[2]) + t
                    is_corrected = True
                    error_num += 1
                    line = line.replace(error_span, correct_span)
        if is_corrected:
            sentence_num += 1
        pred_texts.append(line)
        
    return pred_texts


if __name__ == "__main__":
    r = gec_pattern_correct(['孙中山辞职后不再过问政治，决心尽瘁社会上事业，开始着手社会革命。', '急哦哦i阿斯顿'])
    print(r)
