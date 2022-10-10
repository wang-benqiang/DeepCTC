
import pypinyin
from pypinyin import pinyin
from tqdm import tqdm
import json

yunmuDict = {'a': '1', 'o': '2', 'e': '3', 'i': '4',
             'u': '5', 'v': '6', 'ai': '7', 'ei': '7',
             'ui': '8', 'ao': '9', 'ou': 'A', 'iou': 'B',  # 有：you->yiou->iou->iu
             'ie': 'C', 've': 'D', 'er': 'E', 'an': 'F',
             'en': 'G', 'in': 'H', 'un': 'I', 'vn': 'J',  # 晕：yun->yvn->vn->ven
             'ang': 'F', 'eng': 'G', 'ing': 'H', 'ong': 'K'}

shengmuDict = {'b': '1', 'p': '2', 'm': '3', 'f': '4',
               'd': '5', 't': '6', 'n': '7', 'l': '7',
               'g': '8', 'k': '9', 'h': 'A', 'j': 'B',
               'q': 'C', 'x': 'D', 'zh': 'E', 'ch': 'F',
               'sh': 'G', 'r': 'H', 'z': 'E', 'c': 'F',
               's': 'G', 'y': 'I', 'w': 'J', '0': '0'}

shapeDict = {'⿰': '1', '⿱': '2', '⿲': '3', '⿳': '4', '⿴': '5',  # 左右结构、上下、左中右、上中下、全包围
                  '⿵': '6', '⿶': '7', '⿷': '8', '⿸': '9', '⿹': 'A',  # 上三包、下三包、左三包、左上包、右上包
                  '⿺': 'B', '⿻': 'C', '0': '0'}  # 左下包、镶嵌、独体字：0

strokesDict = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A',
               11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K',
               21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U',
               31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 0: '0'}

hanziStrokesDict = {}  # 汉子：笔画数
hanziStructureDict = {}  # 汉子：形体结构


def getSoundCodes(words):

    shengmuStrs = pinyin(words, style=pypinyin.INITIALS, heteronym=False, strict=False)
    yunmuStrFullStricts = pinyin(words, style=pypinyin.FINALS_TONE3, heteronym=False, strict=True)
    soundCodes = []
    for shengmuStr0, yunmuStrFullStrict0 in zip(shengmuStrs, yunmuStrFullStricts):
        res = []
        shengmuStr = shengmuStr0[0]
        yunmuStrFullStrict = yunmuStrFullStrict0[0]

        if shengmuStr not in shengmuDict:
            shengmuStr = '0'

        yindiao = '0'
        if yunmuStrFullStrict[-1] in ['1','2','3','4']:
            # 音调、韵母
            yindiao = yunmuStrFullStrict[-1]
            yunmuStrFullStrict = yunmuStrFullStrict[:-1]

        if yunmuStrFullStrict in yunmuDict:
            #声母，韵母辅音补码，韵母，音调
            res.append(yunmuDict[yunmuStrFullStrict])
            res.append(shengmuDict[shengmuStr])
            res.append('0')
        elif len(yunmuStrFullStrict)>1:
            res.append(yunmuDict[yunmuStrFullStrict[1:]])
            res.append(shengmuDict[shengmuStr])
            res.append(yunmuDict[yunmuStrFullStrict[0]])
        else:
            res.append('0')
            res.append(shengmuDict[shengmuStr])
            res.append('0')
            
        res.append(yindiao)
        soundCodes.append(res)

    return soundCodes


# soundCode=['2', '8', '5', '2']
def computeSoundCodeSimilarity(soundCode1, soundCode2):
    featureSize = len(soundCode1)
    wights = [0.4, 0.5, 0.1, 0.00]
    multiplier = []
    for i in range(featureSize):
        if soundCode1[i] == soundCode2[i]:
            multiplier.append(1)
        else:
            multiplier.append(0)
    soundSimilarity = 0
    for i in range(featureSize):
        soundSimilarity += wights[i]*multiplier[i]
    return soundSimilarity


def get_char_pronounce_confusion():

    a = getSoundCodes('的地')
    b = computeSoundCodeSimilarity(*a)
    print(b)
    return b


class CharHelper:
    def __init__(self) -> None:
        self.all_chars =  open('src/zh_data/generaral_char.txt', 'r', encoding='utf8').read()
        self.char_pronounce_code = {
            char:getSoundCodes(char)[0] for char in self.all_chars
        }

    def get_char_pronounce_confusion(self):
        self.char_pronounce_confusion = {}
        for src_char, src_code in tqdm(self.char_pronounce_code.items(),  total=len(self.char_pronounce_code)):

            self.char_pronounce_confusion[src_char] = []
            for trg_char, trg_code in self.char_pronounce_code.items():
                if trg_char != src_char:
                    sim = computeSoundCodeSimilarity(src_code, trg_code)
                    if sim > 0.2:
                        self.char_pronounce_confusion[src_char].append([trg_char, sim])
         
    
        
        for src_char, src_code in tqdm(self.char_pronounce_confusion.items(), total=len(self.char_pronounce_code)):
            self.char_pronounce_confusion[src_char] = sorted(self.char_pronounce_confusion[src_char], key=lambda x:x[1], reverse=True)

        return self.char_pronounce_confusion
    
    
if __name__ == '__main__':
    
    # getSoundCodes('爽酸')
    a = CharHelper()
    r = a.get_char_pronounce_confusion()
    

    json.dump(r, open('char_pronounce_confusion4.json', 'w', encoding='utf8'), ensure_ascii=False, indent=2)
    
    # fp = open('char_pronounce_confusion2.json', 'r', encoding='utf8')
    # a = json.load(fp)
    # print('e')
