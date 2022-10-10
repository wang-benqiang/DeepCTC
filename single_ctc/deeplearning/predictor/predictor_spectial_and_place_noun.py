#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from functools import reduce
# import onnxruntime as ort
import torch
import numpy as np
# from src import logger
from src.deeplearning.modeling.modeling_special_and_place import ModelingSpectialAndPlace
# from src.utils.data_helper import (SPACE_SIGNS, replace_punc_for_bert)
# from src.deeplearning.tokenizer.bert_tokenizer import CustomBertTokenizer
# from configs.ctc_conf import CtcConf
from src.utils.data_helper import SPACE_SIGNS, replace_punc_for_bert
from transformers import BertTokenizer

PAD = "[PAD]"
UNK = "@@UNKNOWN@@"
def get_token_action(token, index, prob, sugg_token):
    """Get lost of suggested actions for token."""
    # cases when we don't need to do anything
    if sugg_token in [UNK, PAD, '$KEEP','$place_KEEP','$special_KEEP','[REPLACE_UNK]','$O']:
        return None

    if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') or sugg_token == '$DELETE':
        start_pos = index
        end_pos = index + 1
    elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
        start_pos = index + 1
        end_pos = index + 1

    if sugg_token == "$DELETE":
        sugg_token_clear = ""
    elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
        sugg_token_clear = sugg_token[:]
    else:
        try:
            sugg_token_clear = sugg_token[sugg_token.index('_') + 1:]
        except:
            print(1)

    return start_pos - 1, end_pos - 1, sugg_token_clear, prob

def convert_using_case(token, smart_action):
    if not smart_action.startswith("$TRANSFORM_CASE_"):
        return token
    if smart_action.endswith("LOWER"):
        return token.lower()
    elif smart_action.endswith("UPPER"):
        return token.upper()
    elif smart_action.endswith("CAPITAL"):
        return token.capitalize()
    elif smart_action.endswith("CAPITAL_1"):
        return token[0] + token[1:].capitalize()
    elif smart_action.endswith("UPPER_-1"):
        return token[:-1].upper() + token[-1]
    else:
        return token
def get_verb_form_dicts():
    path_to_dict = os.path.join("verb-form-vocab.txt")
    encode, decode = {}, {}
    with open(path_to_dict, encoding="utf-8") as f:
        for line in f:
            words, tags = line.split(":")
            word1, word2 = words.split("_")
            tag1, tag2 = tags.split("_")
            decode_key = f"{word1}_{tag1}_{tag2.strip()}"
            if decode_key not in decode:
                encode[words] = tags
                decode[decode_key] = word2
    return encode, decode
# ENCODE_VERB_DICT, DECODE_VERB_DICT = get_verb_form_dicts()
# def decode_verb_form(original):
#     return DECODE_VERB_DICT.get(original)

# def convert_using_verb(token, smart_action):
#     key_word = "$TRANSFORM_VERB_"
#     if not smart_action.startswith(key_word):
#         raise Exception(f"Unknown action type {smart_action}")
#     encoding_part = f"{token}_{smart_action[len(key_word):]}"
#     decoded_target_word = decode_verb_form(encoding_part)
#     return decoded_target_word


def convert_using_split(token, smart_action):
    key_word = "$TRANSFORM_SPLIT"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    target_words = token.split("-")
    return " ".join(target_words)


def convert_using_plural(token, smart_action):
    if smart_action.endswith("PLURAL"):
        return token + "s"
    elif smart_action.endswith("SINGULAR"):
        return token[:-1]
    else:
        raise Exception(f"Unknown action type {smart_action}")
def apply_reverse_transformation(source_token, transform):
    if transform.startswith("$TRANSFORM"):
        # deal with equal
        if transform == "$KEEP":
            return source_token
        # deal with case
        if transform.startswith("$TRANSFORM_CASE"):
            return convert_using_case(source_token, transform)
        # deal with verb
        if transform.startswith("$TRANSFORM_VERB"):
            return convert_using_verb(source_token, transform)
        # deal with split
        if transform.startswith("$TRANSFORM_SPLIT"):
            return convert_using_split(source_token, transform)
        # deal with single/plural
        if transform.startswith("$TRANSFORM_AGREEMENT"):
            return convert_using_plural(source_token, transform)
        # raise exception if not find correct type
        raise Exception(f"Unknown action type {transform}")
    else:
        return source_token
def replace_merge_transforms(tokens):
    if all(not x.startswith("$MERGE_") for x in tokens):
        return tokens

    target_line = " ".join(tokens)
    target_line = target_line.replace(" $MERGE_HYPHEN ", "-")
    target_line = target_line.replace(" $MERGE_SPACE ", "")
    return target_line.split()

def get_target_sent_by_edits(source_tokens, edits):
    target_tokens = source_tokens[:]
    shift_idx = 1
    for edit in edits:
        start, end, label, _ = edit
        target_pos = start + shift_idx
        source_token = target_tokens[target_pos] \
            if len(target_tokens) > target_pos >= 0 else ''
        if label == "":
            del target_tokens[target_pos]
            shift_idx -= 1
        elif start == end:
            word = label.replace("$APPEND_", "")
            target_tokens[target_pos: target_pos] = [word]
            shift_idx += 1
        elif label.startswith("$TRANSFORM_"):
            word = apply_reverse_transformation(source_token, label)
            if word is None:
                word = source_token
            target_tokens[target_pos] = word
        elif start == end - 1:
            word = label.replace("$REPLACE_", "")
            target_tokens[target_pos] = word
        elif label.startswith("$MERGE_"):
            target_tokens[target_pos + 1: target_pos + 1] = [label]
            shift_idx += 1

    return replace_merge_transforms(target_tokens)



class PredictorSpectialAndPlace:
    def __init__(
            self,
            pretrained_model_dir,
            info_dict,
            use_cuda=True,
            onnx_mode=False,
            cuda_id=0
    ):
        self.info_dict=info_dict
        self.pretrained_model_dir = pretrained_model_dir
        self.use_cuda = use_cuda
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_dir)
        self.cuda_id = cuda_id
        self.cuda_id = cuda_id
        self.onnx_mode=onnx_mode
        self.model = self.load_model()

    def load_model(self):
        """
        加载模型 & 放置到 GPU 中（单卡）
        """

        # set to device to the first cuda
        device = torch.device("cpu" if not self.use_cuda else "cuda:" + str(self.cuda_id))
        model = ModelingSpectialAndPlace(bert_dir=self.pretrained_model_dir,
                          num_tags=len(self.info_dict['id2ent']), num_dtags=len(self.info_dict['id2dtag']))

        # logger.info(f'Load ckpt from {self.pretrained_model_dir}')
        model.load_state_dict(torch.load(os.path.join(self.pretrained_model_dir,'model.pt'), map_location=torch.device('cpu')), strict=True)
        model.eval()
        model.to(device)
        if self.use_cuda and torch.cuda.is_available():
            model = model.half()
        return model

    def tokenize_inputs(self, texts, return_tensors=None):

        cls_id, sep_id, pad_id, unk_id = self.tokenizer.vocab['[CLS]'], self.tokenizer.vocab[
            '[SEP]'], self.tokenizer.vocab['[PAD]'], self.tokenizer.vocab['[UNK]']
        input_ids, attention_mask, token_type_ids, length = [], [], [], []
        max_len = max([len(text) for text in texts]) + 2  # 注意+2

        for text in texts:
            true_input_id = [self.tokenizer.vocab.get(
                c, unk_id) for c in text][:max_len - 2]
            pad_len = (max_len - len(true_input_id) - 2)
            input_id = [cls_id] + true_input_id + [sep_id] + [pad_id] * pad_len

            a_mask = [1] * (len(true_input_id) + 2) + [0] * pad_len
            token_type_id = [0] * max_len
            input_ids.append(input_id)
            attention_mask.append(a_mask)
            token_type_ids.append(token_type_id)
            length.append(sum(a_mask))

        if return_tensors == 'pt':
            return {'input_ids': torch.LongTensor(input_ids),
                    'attention_mask': torch.LongTensor(attention_mask),
                    'token_type_ids': torch.LongTensor(token_type_ids),
                    'length': torch.LongTensor(length)}
        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'length': length}

    def span_decode(self,start_logits, end_logits, id2ent,threshold=0.4):
        predict_entities = []
        start_logits = np.array(torch.softmax(torch.FloatTensor(start_logits), -1))
        end_logits = np.array(torch.softmax(torch.FloatTensor(end_logits), -1))
        end_pred = np.argmax(end_logits, -1)


        # for idx, e_type in enumerate(end_pred):
        #     if e_type == 2:
        #         start_logits[idx][1] = start_logits[idx][1] - 0 * end_logits[idx][2]
        #     if e_type == 1:
        #         start_logits[idx][1] = start_logits[idx][1] + 0 * end_logits[idx][1]
        start_pred = np.argmax(start_logits, -1)
        start_pred_logits=np.max(start_logits, -1)
        # print(id2ent)
        for s,l in zip(start_pred.tolist(),start_pred_logits.tolist()):
        
            predict_entities.append([id2ent[s_type] if l_num>threshold else id2ent[1] for s_type,l_num in zip(s,l)])

        return predict_entities

    @torch.no_grad()
    def predict(
            self,
            texts,
            batch_size=32,
            return_topk=1,
            threshold=0.4
    ):
        """ seq2label

        Args:
            texts (list):
            ignore_idx_list: [[12], [13,14]]
        Returns:
            List[tuple]: [('你','$KEEP'), ('好', '$DELETE')],
        """

        outputs = []
        if isinstance(texts, str):
            texts = [texts]

        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]

            # 将一些bert词典中不存在的 中文符号 转换为 英文符号
            # 加空格让bert对英文分字
            batch_char_based_texts = [
                ['$START']+[i for i in replace_punc_for_bert(text)]
                for text in batch_texts
            ]
            # inputs = self.tokenize_inputs(
            #     batch_char_based_texts, return_tensors='pt')
            max_len = max([len(text) for text in batch_char_based_texts]) + 2

            inputs=self.tokenize_inputs(batch_char_based_texts,return_tensors='pt')

            # inputs['input_ids'][..., 1] = 1  # 把 始 换成 [unused1]
            if not self.onnx_mode:
                if self.use_cuda and torch.cuda.is_available():
                    inputs['input_ids'] = inputs['input_ids'].cuda()
                    inputs['attention_mask'] = inputs['attention_mask'].cuda()
                    inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

                d_preds, c_preds = self.model(
                    token_ids=inputs['input_ids'],
                    attention_masks=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids'],
                )
            else:
                d_preds, c_preds = self.model.run(None, {
                    'input_ids': inputs['input_ids'].numpy(),
                    'attention_mask': inputs['attention_mask'].numpy(),
                    'token_type_ids': inputs['token_type_ids'].numpy(),
                })[0]
                c_preds = torch.from_numpy(c_preds)





            c_logits = c_preds.cpu().numpy()[:,1:1 + max_len,:]
            d_logits = d_preds.cpu().numpy()[:,1:1 + max_len,:]

            decode_entities = self.span_decode(c_logits, d_logits, self.info_dict['id2ent'],threshold)
            [[decode_entities[text_idx].insert(i, '$O') for i, v in enumerate(batch_texts[text_idx]) if v in SPACE_SIGNS] for text_idx in range(len(batch_texts))]
            [[batch_char_based_texts[text_idx].insert(i+1, v) for i, v in enumerate(batch_texts[text_idx]) if v in SPACE_SIGNS] for text_idx in range(len(batch_texts))]
            for index in range(len(decode_entities)):
                edits = []
                for entity_index in range(len(batch_char_based_texts[index][1:])):
                    token = batch_char_based_texts[index][entity_index+1]
                    # skip if there is no error
                    # if decode_entities[i][1] == 1:
                    #     continue

                    sugg_token = decode_entities[index][entity_index]
                    action = get_token_action(token, entity_index,1,sugg_token)
                    if not action:
                        continue

                    edits.append(action)
                outputs.append(['$START']+get_target_sent_by_edits(batch_char_based_texts[index][1:], edits))


            # outputs.extend(decode_entities)
            # for i in range(len(decode_entities)):
            #     token = sent_tokens[i]
            #     # skip if there is no error
            #     if decode_entities[i][1] == 1:
            #         continue
            #
            #     sugg_token = decode_entities[i][0]
            #     action = get_token_action(token, i, 1,sugg_token)
            #     if not action:
            #         continue
            #
            #     edits.append(action)
            # all_results.append([START_TOKEN]+get_target_sent_by_edits(sent_tokens[1:], edits))
            # decode_result.append(decode_entities)
        outputs=[''.join(i[1:]) for i in outputs]
        return outputs




if __name__ == '__main__':
    import os

    os.chdir('/root/api-yjy-gen-corrector')
    info_dict = {}
    with open('/root/places_ctc/project/gector/vocab/ctc_correct_tags.txt', 'r', encoding='utf-8') as f:
        ent2id = f.readlines()
        ent2id = {j.strip(): i for i, j in enumerate(ent2id)}

    with open('/root/places_ctc/project/gector/vocab/d_tags.txt', 'r', encoding='utf-8') as f:
        dtag2id = f.readlines()
        dtag2id = {j.strip(): i for i, j in enumerate(dtag2id)}

    info_dict['id2ent'] = {ent2id[key]: key for key in ent2id.keys()}
    info_dict['id2dtag'] = {dtag2id[key]: key for key in dtag2id.keys()}

    predictor = PredictorSpectialAndPlace(pretrained_model_dir='/root/places_ctc/project/gector/out/macbertcsc_fineturn_fp16_ls_ce_span/checkpoint-24204',
                                          info_dict=info_dict,
                                          use_cuda=True,
                                          cuda_id=0)
    # texts = ['今天今天的天气真不呀,', '我知道知道这件事情']

    r = predictor.predict(['就天山山鹿侵使面而言,主要是第四纪期间强大的冰水作用侵蚀而成的','就天山山鹿侵使面而言,主要是第四纪期间强大的冰水做用侵蚀而成的'])
    # r = [[(i[0], i[1][0][0]) for i in ele] for ele in r]
    print(r)


