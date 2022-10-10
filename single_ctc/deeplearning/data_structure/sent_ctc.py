class CscChar:
    def __init__(self, src_char, trg_char_prob_list):
        """[summary]

        Args:
            src_char ([type]): 尼
            trg_char_prob_list ([type]): [(牛,0.9), (呢, 0.1)]
        """
        self.src_char = src_char
        self.trg_char_prob_list = trg_char_prob_list
    
    
    def __str__(self):
        return self.trg_char_prob_list[0][0]

    
    
class CscText:
    def __init__(self, outputs):
        self._v = [CscChar(src, trg_char_prob_list)
                   for (src, trg_char_prob_list) in outputs]
    @property
    def src_text(self):
        return ''.join([i.src_char for i in self._v])
    @property
    def trg_text(self):
        return ''.join([i.trg_char_prob_list[0][0] for i in self._v])
    def __getitem__(self, item) -> CscChar:
        return self._v[item]

    def __len__(self) -> int:
        return len(self._v)
    
    


if __name__ == '__main__':
    a  = [ [('我',[('我',1)]) ],
           [('她',[('我',1)]), ('说', [('说', 1)]) ]
           
           
    ]
    
    
    texts = [ CscText(i) for i in a]
    print('end')