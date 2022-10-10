
import torch
from src.deeplearning.modeling.modeling_lm import GPT2LMHeadModel
from src.utils.data_helper import replace_punc_for_bert
from transformers import BertTokenizer


class PredictorLm:
    def __init__(self,
                 pretrained_model_dir,
                 use_cuda=True,
                 cuda_id=None):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model_dir)
        self.use_cuda = use_cuda
        self.cuda_id = cuda_id
        if self.use_cuda and torch.cuda.is_available():
            if cuda_id is not None:
                torch.cuda.set_device(cuda_id)
            self.model.cuda()
            self.model.half()
        self.model.eval()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='mean')

    @torch.no_grad()
    def __call__(self, texts, max_length=128, batch_size=32,
                 return_topk=2):
        if isinstance(texts, str):
            texts = [texts]
        loss_list = []
        texts = [replace_punc_for_bert(text)[:max_length-2] for text in texts]
        outputs = []
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx+batch_size]
            true_seq_lens = [len(text) for text in batch_texts]
            inputs = self.list_tokenizer(batch_texts, max_len=max_length)
            inputs['input_ids'] = torch.LongTensor(inputs['input_ids'])
            inputs['attention_mask'] = torch.LongTensor(
                inputs['attention_mask'])
            inputs['token_type_ids'] = torch.LongTensor(
                inputs['token_type_ids'])
            if self.use_cuda and torch.cuda.is_available():
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()
                inputs['token_type_ids'] = inputs['token_type_ids'].cuda()

            preds = self.model(**inputs).logits
            preds = preds
            # preds = torch.softmax(preds[:, 1:, :], dim=-1)  # 从cls后面开始
            preds = torch.softmax(preds[:, 0:, :], dim=-1)  # 从cls后面开始
            recall_top_k_probs, recall_top_k_ids = preds.topk(
                k=return_topk, dim=-1, largest=True, sorted=True)
            recall_top_k_probs = recall_top_k_probs.tolist()
            recall_top_k_ids = recall_top_k_ids.tolist()
            recall_top_k_chars = [[self.tokenizer.convert_ids_to_tokens(
                char_level) for char_level in sent_level] for sent_level in recall_top_k_ids]
            # batch_texts = [ ['']+list(t)[1:] for t in batch_texts] # 占位符
            # batch_texts = [list(t) for t in batch_texts]  # 占位符
            batch_outputs = [list(zip(' '+text+' ', top_k_char, top_k_prob)) for text, top_k_char, top_k_prob in zip(
                batch_texts, recall_top_k_chars, recall_top_k_probs)]
            outputs.extend(batch_outputs)
        return outputs

        return loss_list

    def list_tokenizer(self, sentence_list, max_len):
        single_flag = 0
        if isinstance(sentence_list, str):
            single_flag = 1
            sentence_list = [sentence_list]
        sentence_list_limit = [sent[:max_len - 2] for sent in sentence_list]
        token_list = [[self.tokenizer.vocab.get(wd, self.tokenizer.vocab['[UNK]']) for wd in sent]
                      for sent in sentence_list_limit]
        token_list_format = [[self.tokenizer.vocab["[CLS]"]] + sent +
                             [self.tokenizer.vocab["[SEP]"]] for sent in token_list]
        sentence_list_pad = [sent + [self.tokenizer.vocab["[PAD]"]]
                             * (max_len - len(sent)) for sent in token_list_format]
        attention_mask = [
            [1] * len(sent) + [0] * (max_len - len(sent)) for sent in token_list_format

        ]
        seg_idx = [[0] * max_len for i in range(len(sentence_list_pad))]
        if single_flag == 1:
            return {
                "input_ids": sentence_list_pad[0],
                "token_type_ids": seg_idx[0],
                "attention_mask": attention_mask[0]
            }
        else:
            return {
                "input_ids": sentence_list_pad,
                "token_type_ids": seg_idx,
                "attention_mask": attention_mask
            }


if __name__ == '__main__':
    pretrained_model_dir = 'pretrained_model/gpt2_cn_cluesmall'
    predictor = PredictorLm(pretrained_model_dir='pretrained_model/gpt2_cn_cluesmall',
                            use_cuda=False)
    texts = ['我将会在公司随机挑选几个同意和我一起出差。']

    r = predictor(texts)
    print(1)
    print(r)
