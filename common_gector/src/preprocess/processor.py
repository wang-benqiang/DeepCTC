import os
import logging
from transformers import BertTokenizer

logger = logging.getLogger(__name__)



class InputExample:
    def __init__(self,
                 set_type,
                 text,
                 labels=None,
                 pseudo=None,
                 distant_labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels
        self.pseudo = pseudo
        self.distant_labels = distant_labels


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids





class Feature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 start_ids=None,
                 end_ids=None,
                 pseudo=None):
        super(Feature, self).__init__(token_ids=token_ids,
                                          attention_masks=attention_masks,
                                          token_type_ids=token_type_ids)
        self.start_ids = start_ids
        self.end_ids = end_ids
        # self.ide_ids = ide_ids
        # pseudo
        self.pseudo = pseudo






def read_txt(path):
    with open(path,'r',encoding='utf-8') as f:
        data=f.readlines()
    return data



def extract_tags(tags):
    op_del = 'SEPL__SEPR'

    labels = [x.split(op_del) for x in tags]

    comlex_flag_dict = {}
    # get flags
    for i in range(5):
        idx = i + 1
        comlex_flag_dict[idx] = sum([len(x) > idx for x in labels])

    detect_tags = ["CORRECT" if label[0] == "$KEEP" else "INCORRECT" for label in labels]

    return labels, detect_tags, comlex_flag_dict

def convert_example(example, tokenizer: BertTokenizer,
                         max_seq_len, ent2id,dtag2id,set_type):

    line = example.strip("\n")


    tokens_and_tags = [pair.rsplit('SEPL|||SEPR', 1)
                       for pair in line.split(' ')]
    try:
        tokens = [token for token, tag in tokens_and_tags]
        tags = [tag for token, tag in tokens_and_tags]
    except ValueError:
        tokens = [token[0] for token in tokens_and_tags]
        tags = None

    if tokens and tokens[0] != '$START':
        tokens = ['$START'] + tokens

    # words = [x.text for x in tokens]
    if max_seq_len is not None:
        tokens = tokens[:max_seq_len]
        tags = None if tags is None else tags[:max_seq_len]
    labels, detect_tags, comlex_flag_dict=extract_tags(tags)


    start_ids, end_ids= None, None

    if set_type == 'train':
        start_ids = []
        end_ids = []

        for _ent in labels:
            try:
                ent_type = ent2id[_ent[0]]
            except:
                ent_type= ent2id['@@UNKNOWN@@']
            # ent_start = _ent[-1]
            # ent_end = ent_start + len(_ent[1]) - 1

            start_ids.append(ent_type)
        for _dent in detect_tags:

            ent_type = dtag2id[_dent]
            # ent_start = _ent[-1]
            # ent_end = ent_start + len(_ent[1]) - 1

            end_ids.append(ent_type)


        assert len(start_ids) == len(end_ids)
        if len(start_ids) > max_seq_len - 2:
            start_ids = start_ids[:max_seq_len - 2]
            end_ids = end_ids[:max_seq_len - 2]
            # ide_ids = ide_ids[:max_seq_len - 2]
        #
        start_ids = [0] + start_ids + [0]
        end_ids = [0] + end_ids + [0]
        # ide_ids= [0] + ide_ids + [0]

        # pad
        if len(start_ids) < max_seq_len:
            pad_length = max_seq_len - len(start_ids)

            start_ids = start_ids + [0] * pad_length  # CLS SEP PAD label都为O
            end_ids = end_ids + [0] * pad_length
            # ide_ids = ide_ids + [0] * pad_length

        assert len(start_ids) == max_seq_len
        assert len(end_ids) == max_seq_len
        # assert len(ide_ids) == max_seq_len

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']



    feature = Feature(token_ids=token_ids,
                          attention_masks=attention_masks,
                          token_type_ids=token_type_ids,
                          start_ids=start_ids,
                          end_ids=end_ids,
                          pseudo=0)

    return feature




def convert_examples_to_features(examples, max_seq_len, bert_dir, ent2id,dtag2id,set_type='train'):

    tokenizer = BertTokenizer(os.path.join(bert_dir, 'vocab.txt'))

    features = []

    logger.info(f'Convert {len(examples)} examples to features')
    # type2id = {x: i for i, x in enumerate(ENTITY_TYPES)}

    for i, example in enumerate(examples):
        feature = convert_example(
            example=example,
            max_seq_len=max_seq_len,
            ent2id=ent2id,
            dtag2id=dtag2id,
            tokenizer=tokenizer,
            set_type=set_type
        )

        if feature is None:
            continue


        features.append(feature)

    return features


if __name__ == '__main__':
    pass
