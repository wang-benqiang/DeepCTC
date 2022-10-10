
import pypinyin
from pypinyin import pinyin
from tqdm import tqdm
import json
from src.soundshapecode.four_corner import FourCornerMethod
import pkg_resources


strokesDictReverse = {'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'A':10,
               'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'I':18, 'J':19, 'K':20,
               'L':21, 'M':22, 'N':23, 'O':24, 'P':25, 'Q':26, 'R':27, 'S':28, 'T':29, 'U':30,
               'V':31, 'W':32, 'X':33, 'Y':34, 'Z':35, '0':0}
fcm = FourCornerMethod()


shapeDict = {'⿰':'1','⿱':'2','⿲':'3','⿳':'4','⿴':'5',#左右结构、上下、左中右、上中下、全包围
                  '⿵':'6','⿶':'7','⿷':'8','⿸':'9','⿹':'A',#上三包、下三包、左三包、左上包、右上包
                  '⿺':'B','⿻':'C', '0':'0'}#左下包、镶嵌、独体字：0

strokesDict = {1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
               11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
               21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U',
               31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z', 0:'0'}


def getHanziStructureDict():
    hanziStructureDict = {}
    structure_filepath = 'utils/sound_shape_code/unihan_structure.txt'
    with open(structure_filepath, 'r', encoding='UTF-8') as f:#文件特征：U+4EFF\t仿\t⿰亻方\n
        for line in f:
            line = line.split()
            if line[2][0] in shapeDict:
                hanziStructureDict[line[1]]=line[2][0]
    return hanziStructureDict



def getHanziStrokesDict():
    hanziStrokesDict = {}
    strokes_filepath = 'src/zh_data/utf8_strokes.txt'
    with open(strokes_filepath, 'r', encoding='UTF-8') as f:#文件特征：
        for line in f:
            line = line.split()
            hanziStrokesDict[line[1]]=line[2]
    return hanziStrokesDict


def getShapeCode(one_chi_word):
    res = []
    structureShape = hanziStructureDict.get(one_chi_word, '0')#形体结构
    res.append(shapeDict[structureShape])
    
    fourCornerCode = fcm.query(one_chi_word)#四角号码（5位数字）
    if fourCornerCode is None:
        res.extend(['0', '0', '0', '0', '0'])
    else:
        res.extend(fourCornerCode[:])
    
    strokes = hanziStrokesDict.get(one_chi_word, '0')#笔画数
    if int(strokes) >35:
        res.append('Z')
    else:
        res.append(strokesDict[int(strokes)])     
    return res


def computeShapeCodeSimilarity(shapeCode1, shapeCode2):#shapeCode=['5', '6', '0', '1', '0', '3', '8']
    featureSize=len(shapeCode1)
    wights=[0.25,0.1,0.1,0.15,0.15,0.15,0.1]
    multiplier=[]
    for i in range(featureSize-1):
        if shapeCode1[i]==shapeCode2[i]:
            multiplier.append(1)
        else:
            multiplier.append(0)
    multiplier.append(1- abs(strokesDictReverse[shapeCode1[-1]]-strokesDictReverse[shapeCode2[-1]])*1.0 / max(strokesDictReverse[shapeCode1[-1]],strokesDictReverse[shapeCode2[-1]]) )
    shapeSimilarity=0
    for i in range(featureSize):
        shapeSimilarity += wights[i]*multiplier[i]
    return shapeSimilarity


hanziStructureDict = getHanziStructureDict()
hanziStrokesDict = getHanziStrokesDict()



class CharHelper:
    def __init__(self) -> None:
        self.all_chars =  open('src/zh_data/generaral_char.txt', 'r', encoding='utf8').read()
        self.char_shape_code = {
            char:getShapeCode(char) for char in self.all_chars
        }
        print('a')
    def get_char_shape_confusion(self):
        self.char_shape_confusion = {}
        for src_char, src_code in tqdm(self.char_shape_code.items(),  total=len(self.char_shape_code)):

            self.char_shape_confusion[src_char] = []
            for trg_char, trg_code in self.char_shape_code.items():
                if trg_char != src_char:
                    sim = computeShapeCodeSimilarity(src_code, trg_code)
                    if sim >= 0.45:
                        self.char_shape_confusion[src_char].append([trg_char, sim])
         
                        
        

        
        for src_char, src_code in tqdm(self.char_shape_confusion.items(), total=len(self.char_shape_code)):
            self.char_shape_confusion[src_char] = sorted(self.char_shape_confusion[src_char], key=lambda x:x[1], reverse=True)

        return self.char_shape_confusion
    
    
if __name__ == '__main__':
    
    # getSoundCodes('爽酸')
    a = CharHelper()
    r = a.get_char_shape_confusion()
    
    print('e')
    json.dump(r, open('char_shape_confusion2.json', 'w', encoding='utf8'), ensure_ascii=False, indent=2)
    