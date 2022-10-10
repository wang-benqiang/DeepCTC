from src.utils.model_utils import GecModel
from transformers import BertTokenizer
import os
from src.utils.functions_utils import load_model_and_parallel, ensemble_vote
import torch
from src.utils.evaluator import span_decode
# import numpy as np
from src.predictor_ctc_seq2edit import PredictorCtcSeq2Edit
import copy
GPU_IDS = "0"
with open('./best_ckpt_path.txt', 'r', encoding='utf-8') as f:
    CKPT_PATH = f.readlines()[0].strip()
midu_ctc_bertdir = "./model/ctc_csc"
# macbert4csc_ctc2021_bertdir = "/root/baselines/ctc_ner/DeepNER/model/macbert4csc"
macbert4csc_ctc_bertdir = "./model/macbert4csc"
# macbert_lg_ctc_bertdir = "/root/baselines/ctc_ner/DeepNER/model/chinese-macbert-large"
START_TOKEN = "$START"
VOCAB_DIR = './model'
PAD = "@@PADDING@@"
UNK = "@@UNKNOWN@@"
SEQ_DELIMETERS = {"tokens": " ",
                  "labels": "SEPL|||SEPR",
                  "operations": "SEPL__SEPR"}
MAX_SEQ_LEN = 128

def get_token_action(token, index, prob, sugg_token):
    """Get lost of suggested actions for token."""
    # cases when we don't need to do anything
    if sugg_token in [UNK, PAD, '$KEEP']:
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
    path_to_dict = os.path.join(VOCAB_DIR, "verb-form-vocab.txt")
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
ENCODE_VERB_DICT, DECODE_VERB_DICT = get_verb_form_dicts()
def decode_verb_form(original):
    return DECODE_VERB_DICT.get(original)

def convert_using_verb(token, smart_action):
    key_word = "$TRANSFORM_VERB_"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    encoding_part = f"{token}_{smart_action[len(key_word):]}"
    decoded_target_word = decode_verb_form(encoding_part)
    return decoded_target_word


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
    shift_idx = 0
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

def base_predict(model, device, info_dict):
    tokenizer = info_dict['tokenizer']
    id2ent = info_dict['id2ent']
    all_results = []
    decode_result=[]
    with torch.no_grad():
        for sent_tokens in info_dict['examples']:
            encode_dict = tokenizer.encode_plus(text=sent_tokens,
                                                max_length=MAX_SEQ_LEN,
                                                is_pretokenized=True,
                                                pad_to_max_length=False,
                                                return_tensors='pt',
                                                return_token_type_ids=True,
                                                return_attention_mask=True)

            model_inputs = {'token_ids': encode_dict['input_ids'],
                            'attention_masks': encode_dict['attention_mask'],
                            'token_type_ids': encode_dict['token_type_ids']}

            for key in model_inputs:
                model_inputs[key] = model_inputs[key].to(device)

            start_logits, end_logits = model(**model_inputs)
            start_logits = start_logits[0].cpu().numpy()[1:1 + len(sent_tokens)]
            end_logits = end_logits[0].cpu().numpy()[1:1 + len(sent_tokens)]

            decode_entities = span_decode(start_logits, end_logits, sent_tokens, id2ent)

            edits = []

            for i in range(len(decode_entities)):
                token = sent_tokens[i]
                # skip if there is no error
                if decode_entities[i][1] == 1:
                    continue

                sugg_token = decode_entities[i][0]
                action = get_token_action(token, i, 1,sugg_token)
                if not action:
                    continue
                edits.append(action)
            all_results.append([START_TOKEN]+get_target_sent_by_edits(sent_tokens[1:], edits))
            decode_result.append(decode_entities)
    return all_results,decode_result



def prepare_info(BERT_DIR):
    info_dict = {}
    with open('./data/labels.txt','r', encoding='utf-8') as f:
    # with open('/root/baselines/track2/data/output_vocabulary/labels.txt','r', encoding='utf-8') as f:
        ent2id=f.readlines()
        ent2id = {j.strip():i for i,j in enumerate(ent2id)}

    with open('./data/d_tags.txt','r', encoding='utf-8') as f:
    # with open('/root/baselines/track2/data/output_vocabulary/d_tags.txt','r', encoding='utf-8') as f:
        dtag2id=f.readlines()
        dtag2id = {j.strip():i for i,j in enumerate(dtag2id)}

    info_dict['id2ent'] = {ent2id[key]: key for key in ent2id.keys()}
    info_dict['id2dtag'] = {dtag2id[key]: key for key in dtag2id.keys()}

    info_dict['tokenizer'] = BertTokenizer(os.path.join(BERT_DIR, 'vocab.txt'))

    return info_dict

def mixed_predict(texts,cged_test,lg_result=True):
    MIX_DIR_LIST_reader=open('best_ckpt_path.txt','r',encoding='utf-8')
    MIX_DIR_LIST=MIX_DIR_LIST_reader.readlines()
    MIX_DIR_LIST_reader.close()
    model_path_list = [x.strip() for x in MIX_DIR_LIST]
    print('model_path_list:{}'.format(model_path_list))


    # all_labels = []
    model_list=[]
    for i, model_path in enumerate(model_path_list):
        if i == 1:
            info_dict=prepare_info(macbert4csc_ctc_bertdir)
            # info_dict=prepare_info(macbert4csc_ctc2021_bertdir)
            info_dict['examples'] = [[START_TOKEN] + list(i) for i in texts]
            model = GecModel(bert_dir=macbert4csc_ctc_bertdir, num_tags=len(info_dict['id2ent']),num_dtags=len(info_dict['id2dtag']))
            print(f'Load model from {model_path}')
            model, device = load_model_and_parallel(model, GPU_IDS, model_path)
            model.eval()
            model_list.append((model,device,info_dict))
        elif i == 2:
            info_dict = prepare_info(macbert4csc_ctc_bertdir)
            info_dict['examples'] = [[START_TOKEN] + list(i) for i in texts]
            # info_dict['examples'] = [[START_TOKEN] + info_dict["tokenizer"].tokenize(i) for i in list(texts)]
            model = GecModel(bert_dir=macbert4csc_ctc_bertdir, num_tags=len(info_dict['id2ent']),num_dtags=len(info_dict['id2dtag']))
            print(f'Load model from {model_path}')
            model, device = load_model_and_parallel(model, GPU_IDS, model_path)
            model.eval()
            model_list.append((model, device, info_dict))
        elif i ==6 :
            info_dict = prepare_info(midu_ctc_bertdir)
            info_dict['examples'] = [[START_TOKEN] + list(i) for i in texts]
            # info_dict['examples'] = [[START_TOKEN] + info_dict["tokenizer"].tokenize(i) for i in list(texts)]
            model = GecModel(bert_dir=midu_ctc_bertdir, num_tags=len(info_dict['id2ent']),num_dtags=len(info_dict['id2dtag']))
            print(f'Load model from {model_path}')
            model, device = load_model_and_parallel(model, GPU_IDS, model_path)
            model.eval()
            model_list.append((model, device, info_dict))
        # elif i==90:
        #     info_dict=prepare_info(macbert_lg_ctc_bertdir)
        #     info_dict['examples'] = [[START_TOKEN] + list(i) for i in texts]
        #     # info_dict['examples'] = [[START_TOKEN] + info_dict["tokenizer"].tokenize(i) for i in list(texts)]
        #     model = GecModel(bert_dir=macbert_lg_ctc_bertdir, num_tags=len(info_dict['id2ent']),num_dtags=len(info_dict['id2dtag']))
        #     print(f'Load model from {model_path}')
        #     model, device = load_model_and_parallel(model, GPU_IDS, model_path)
        #     model.eval()
        #     model_list.append((model, device,info_dict))

        # all_labels.append(labels)
    # labels=[]
    labels=[[START_TOKEN] + list(i) for i in texts]
    lg_model_text = copy.deepcopy(texts)
    for i in range(4):
        all_labels=[]
        if lg_result:
            if labels:
                lg_text = [''.join(i[1:]).replace("##", '').upper() for i in labels]
            else:
                lg_text = lg_model_text
            p1 = PredictorCtcSeq2Edit(
                in_model_dir='model/epoch10,step251,testepochf1_0.3068,devepochf1_0.4571',
                use_cuda=True, cuda_id=int(GPU_IDS))
            r1 = p1._predict(lg_text, return_topk=5)
            all_labels.append([[j[1][0] for j in i] for i in r1])

        for model,device,info_dict in model_list:
            if labels:
                info_dict['examples'] = labels
            _, decode_entities = base_predict(model, device, info_dict)
            all_labels.append([[j[0] for j in i] for i in decode_entities])

        ensemble_labels = ensemble_vote(all_labels)
        all_results=[]
        for label,sent_tokens in zip(ensemble_labels,labels):

            edits = []
            for i in range(len(label)):
                token = sent_tokens[i]
                # skip if there is no error
                if label[i] == "$KEEP":
                    continue

                sugg_token = label[i]
                if sugg_token==' ':
                    print(1)
                # sugg_token = label[i][0]
                action = get_token_action(token, i, 1, sugg_token)
                if not action:
                    continue

                edits.append(action)
            all_results.append([START_TOKEN] + get_target_sent_by_edits(sent_tokens[1:], edits))
        labels=all_results

    tokenizer = BertTokenizer(os.path.join(macbert4csc_ctc_bertdir, 'vocab.txt'))
    for i,j in zip(labels,cged_test):
        if "[UNK]" in i:
            target=tokenizer.convert_tokens_to_ids(list(j))
            while True:
                try:

                    i[i.index('[UNK]')] = j[target.index(101)]
                    target[target.index(101)] = 0
                except:
                    break
        else:
            print(i)

    result=[''.join(i[1:]).replace("##",'').upper() for i in labels]


    return result



def ner_ctc_predict(texts,cged_test):
    info_dict=prepare_info(macbert4csc_ctc_bertdir)
    model = GecModel(bert_dir=macbert4csc_ctc_bertdir, num_tags=len(info_dict['id2ent']))
    model, device = load_model_and_parallel(model, GPU_IDS, CKPT_PATH)
    model.eval()
    info_dict['examples'] = [[START_TOKEN] + info_dict["tokenizer"].tokenize(i) for i in texts]

    for i in range(4):
        examples,_ = base_predict(model, device, info_dict)
        info_dict['examples']=examples

    for i,j in zip(examples,cged_test):
        if "[UNK]" in i:
            target=info_dict["tokenizer"].tokenize(j)
            while True:
                try:
                    i[i.index('[UNK]')]=j[target.index('[UNK]')]
                    target[target.index('[UNK]')]=j[target.index('[UNK]')]
                except:
                    break
        else:
            print(i)

    result=[''.join(i[1:]).replace("##",'').upper()  for i in examples]

    return result
