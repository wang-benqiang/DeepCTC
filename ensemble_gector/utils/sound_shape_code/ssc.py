
import json
import pickle
from typing import List

import pypinyin
from pypinyin import pinyin
from tqdm import tqdm
from utils.data_helper import include_cn


class CharInforHelper:

    structure_filepath = 'utils/sound_shape_code/data/unihan_structure.txt'
    strokes_filepath = 'utils/sound_shape_code/data/utf8_strokes.txt'
    four_corner_fp = 'utils/sound_shape_code/data/four_corner.pkl'

    yunmu_dict = {'a': '1', 'o': '2', 'e': '3', 'i': '4',
                  'u': '5', 'v': '6', 'ai': '7', 'ei': '7',
                  'ui': '8', 'ao': '9', 'ou': 'A', 'iou': 'B',  # 有：you->yiou->iou->iu
                  'ie': 'C', 've': 'D', 'er': 'E', 'an': 'F',
                  'en': 'G', 'in': 'H', 'un': 'I', 'vn': 'J',  # 晕：yun->yvn->vn->ven
                  'ang': 'F', 'eng': 'G', 'ing': 'H', 'ong': 'K'}

    shengmu_dict = {'b': '1', 'p': '2', 'm': '3', 'f': '4',
                    'd': '5', 't': '6', 'n': '7', 'l': '7',
                    'g': '8', 'k': '9', 'h': '4', 'j': 'B',
                    'q': 'C', 'x': 'D', 'zh': 'E', 'ch': 'F',
                    'sh': 'G', 'r': 'H', 'z': 'E', 'c': 'F',
                    's': 'G', 'y': 'I', 'w': 'J', '0': '0'}     # hf, ln '0' represent null

    shape_dict = {'⿰': '1', '⿱': '2', '⿲': '3', '⿳': '4', '⿴': '5',  # 左右结构、上下、左中右、上中下、全包围
                  '⿵': '6', '⿶': '7', '⿷': '8', '⿸': '9', '⿹': 'A',  # 上三包、下三包、左三包、左上包、右上包
                  '⿺': 'B', '⿻': 'C', '0': '0'}  # 左下包、镶嵌、独体字：0

    strokes_dict = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A',
                    11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K',
                    21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U',
                    31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 0: '0'}

    id2stroke = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10,
                 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
                 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30,
                 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, '0': 0}

    "汉字形体结构"
    char_structure_dict = {}

    with open(structure_filepath, 'r', encoding='UTF-8') as f:  # 文件特征：U+4EFF\t仿\t⿰亻方\n
        for line in f:
            line = line.split()
            if line[2][0] in shape_dict:
                char_structure_dict[line[1]] = line[2][0]

    "汉字笔画数"
    char_strokes_dict = {}

    with open(strokes_filepath, 'r', encoding='UTF-8') as f:  # 文件特征：
        for line in f:
            line = line.split()
            char_strokes_dict[line[1]] = line[2]
    "四角编码"
    four_corner_data = pickle.load(open(four_corner_fp, 'rb'))

    @staticmethod
    def query_char_four_corner(char, default=None):
        return CharInforHelper.four_corner_data.get(char, default)





class CharHelper:
    def __init__(self,
                 char_sound_fp='utils/sound_shape_code/data/char2sound.json',
                 char_shape_fp='utils/sound_shape_code/data/char2shape.json',
                 ) -> None:
        # self.all_chars = open(
        #     'src/zh_data/generaral_char.txt', 'r', encoding='utf8').read()
        if char_sound_fp is not None:
            self.char2sound_code = json.load(
                open(char_sound_fp, 'r', encoding='utf8'))
            self.char2shape_code = json.load(
                open(char_shape_fp, 'r', encoding='utf8'))

        self._unk_sound_code = ['0'] * 4
        self._unk_shape_code = ['0'] * 7

    def gen_char_code_jsonfile(self,
                               out_char_sound_fp='utils/sound_shape_code/data/char2sound.json',
                               out_char_shape_fp='utils/sound_shape_code/data/char2shape.json',
                               ):

        

        all_cn_chars = [line.strip().split(' ')[1] for line in open('utils/sound_shape_code/data/utf8_strokes.txt', 'r', encoding='utf8')]

        all_cn_chars = [i for i in all_cn_chars if include_cn(i)]

        char_sound_code = {char: self.get_sound_code(
            char)[0] for char in all_cn_chars}

        char_shape_code = {char: self.get_shape_code(
            char)[0] for char in all_cn_chars}

        
        out_char_sound_fp = open(out_char_sound_fp, 'w', encoding='utf8')

        out_char_shape_fp = open(out_char_shape_fp, 'w', encoding='utf8')
        json.dump(char_sound_code, out_char_sound_fp, ensure_ascii=False, indent=2)
        json.dump(char_shape_code, out_char_shape_fp, ensure_ascii=False, indent=2)

    def get_sound_code(self, text, read_from_json=True) -> List[list]:

        if read_from_json:
            sound_code_list = [self.char2sound_code.get(
                char, self._unk_sound_code) for char in text]
            return sound_code_list

        sound_code_list = []
        shengmu_str_list = pinyin(text, style=pypinyin.INITIALS,
                                  heteronym=False, strict=False)
        yunmu_str_full_stricts_list = pinyin(
            text, style=pypinyin.FINALS_TONE3, heteronym=False, strict=True)

        for shengmu_list, yunmu_list in zip(shengmu_str_list, yunmu_str_full_stricts_list):
            sound_code = []
            shengmu_str = shengmu_list[0]
            yunmu_str = yunmu_list[0]

            if yunmu_str == '':
                "error case"
                return self._unk_sound_code

            if shengmu_str not in CharInforHelper.shengmu_dict:
                shengmu_str = '0'

            yindiao = '0'
            if yunmu_str[-1] in ['1', '2', '3', '4']:
                # 音调、韵母
                yindiao = yunmu_str[-1]
                yunmu_str = yunmu_str[:-1]

            if yunmu_str in CharInforHelper.yunmu_dict:
                # 声母，韵母辅音补码，韵母，音调
                sound_code.append(CharInforHelper.yunmu_dict[yunmu_str])
                sound_code.append(CharInforHelper.shengmu_dict[shengmu_str])
                sound_code.append('0')
            elif len(yunmu_str) > 1:
                sound_code.append(CharInforHelper.yunmu_dict[yunmu_str[1:]])
                sound_code.append(CharInforHelper.shengmu_dict[shengmu_str])
                sound_code.append(CharInforHelper.yunmu_dict[yunmu_str[0]])
            else:
                sound_code.append('0')
                sound_code.append(CharInforHelper.shengmu_dict[shengmu_str])
                sound_code.append('0')

            sound_code.append(yindiao)
            sound_code_list.append(sound_code)

        return sound_code_list

    def get_shape_code(self, text, read_from_json=True) -> List[list]:

        if read_from_json:
            shape_code_list = [self.char2shape_code.get(
                char, self._unk_shape_code) for char in text]
            return shape_code_list

        shape_code_list = []

        for char in text:
            shape_code = []
            structure_shape = CharInforHelper.char_structure_dict.get(
                char, '0')  # 形体结构
            shape_code.append(CharInforHelper.shape_dict[structure_shape])

            fourCornerCode = CharInforHelper.query_char_four_corner(
                char)  # 四角号码（5位数字）
            if fourCornerCode is None:
                shape_code.extend(['0', '0', '0', '0', '0'])
            else:
                shape_code.extend(fourCornerCode[:])

            strokes = CharInforHelper.char_strokes_dict.get(char, '0')  # 笔画数
            if int(strokes) > 35:
                shape_code.append('Z')
            else:
                shape_code.append(CharInforHelper.strokes_dict[int(strokes)])
            shape_code_list.append(shape_code)
        return shape_code_list

    def compute_shape_similarity_2char(self,
                                       char1,
                                       char2,
                                       weights=[0.25, 0.1, 0.1,
                                                0.15, 0.15, 0.15, 0.1],
                                       ) -> float:
        # shapeCode=['5', '6', '0', '1', '0', '3', '8']
        shape_code1, shape_code2 = self.get_shape_code(char1+char2)
        if shape_code1 == self._unk_shape_code or shape_code2 == self._unk_shape_code:
            return 0
        feature_score = [weights[i] if e1 == e2 else 0 for i, (e1, e2) in enumerate(
            zip(shape_code1[:-1], shape_code2[:-1]))]
        # 笔画特殊计算
        id2stroke = CharInforHelper.id2stroke
        stroke_score = (1 - abs(id2stroke[shape_code1[-1]]-id2stroke[shape_code2[-1]])*1.0 / (max(
            id2stroke[shape_code1[-1]], id2stroke[shape_code2[-1]])) + 1) * weights[-1] # +1 lpls smoothing
        feature_score.append(stroke_score)
        shape_similarity = sum(feature_score)

        return shape_similarity

    def compute_sound_similarity_2char(self,
                                       char1,
                                       char2,
                                       weights=[0.4, 0.5, 0.1, 0.00],
                                       ) -> float:
        # shapeCode=['5', '6', '0', '1', '0', '3', '8']
        sound_code1, sound_code2 = self.get_sound_code(char1+char2)
        feature_score = [weights[i] if e1 == e2 else 0 for i, (e1, e2) in enumerate(
            zip(sound_code1[:-1], sound_code2[:-1]))]

        sound_similarity = sum(feature_score)

        return sound_similarity

    def compute_similarity_2char(self, char1, char2, sound_similarity_weight=0.6):
        sound_similarity = self.compute_sound_similarity_2char(char1, char2)
        shape_similarity = self.compute_shape_similarity_2char(char1, char2)
        similarity = sound_similarity_weight * sound_similarity + \
            (1-sound_similarity_weight) * shape_similarity
        return similarity

    def gen_char_confusion_jsonfile(self, 
                                out_char_confusion_fp='utils/sound_shape_code/data/char2confusion_60percent.json'
                                ):
        
        all_simple_cn_chars = json.load(open('utils/sound_shape_code/data/simple_cn_chars.json', 'r', encoding='utf8'))
        

        char2confusion = {i:[] for i in all_simple_cn_chars}
        for char1 in tqdm(all_simple_cn_chars, total=len(all_simple_cn_chars)):
            # error continue
            if self.get_sound_code(char1) == self._unk_sound_code:
                continue
            # search confusion chars
            for char2 in all_simple_cn_chars:
                if char1 == char2:
                    continue
                similarity = self.compute_similarity_2char(char1, char2)
                if similarity > 0.6:
                    char2confusion[char1].append([char2, round(similarity, 4)])
        
        # sort confusion by similarity
        
        for char, confusion_chars in char2confusion.items():
            char2confusion[char] = sorted(confusion_chars, key=lambda ele:ele[1], reverse=True)
            
        out_char_confusion_fp =open(out_char_confusion_fp, 'w', encoding='utf8')
        
        json.dump(char2confusion, out_char_confusion_fp, ensure_ascii=False, indent=2)
        
        return char2confusion
                
            
        
        
    def gen_cn_chars_pronounce_confusion(self):
            
        char_pronounce_code = {
            char: self.get_sound_code(char)[0] for char in self.all_chars
        }

        self.char_pronounce_confusion = {}
        for src_char, src_code in tqdm(char_pronounce_code.items(),  total=len(char_pronounce_code)):

            self.char_pronounce_confusion[src_char] = []
            for trg_char, trg_code in char_pronounce_code.items():
                if trg_char != src_char:
                    sim = computeSoundCodeSimilarity(src_code, trg_code)
                    if sim > 0.2:
                        self.char_pronounce_confusion[src_char].append(
                            [trg_char, round(sim, 4)])

        for src_char, src_code in tqdm(self.char_pronounce_confusion.items(), total=len(char_pronounce_code)):
            self.char_pronounce_confusion[src_char] = sorted(
                self.char_pronounce_confusion[src_char], key=lambda x: x[1], reverse=True)

        return self.char_pronounce_confusion


if __name__ == '__main__':
    ch = CharHelper()
    # ch.get_sound_code('呣')
    ch.gen_char_confusion_jsonfile()

    # a = pinyin('是', style=pypinyin.INITIALS,
    #            heteronym=False, strict=False)
    # print(a)

    # h = CharHelper()
    # r = h.compute_shape_similarity_2char('霜', '霜')
    # r = h.compute_sound_similarity_2char('霜', '上')
    # print(r)
