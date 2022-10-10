import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class SeqClsModel(nn.Module):
    def __init__(self, num_labels, model_checkpoint):
        nn.Module.__init__(self)
        self.num_labels = num_labels
        self.back_bone = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                            num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.back_bone(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
        return output

    def train_step(self, input_ids, attention_mask, labels):
        loss = self(input_ids, attention_mask, labels).loss
        return {'loss': loss}

    def evaluate_step(self, input_ids, attention_mask, labels):
        pred = self(input_ids, attention_mask, labels).logits
        pred = torch.max(pred, dim=-1)[1]
        return {'pred': pred, 'target': labels}

def cls_correction(texts):

    model=SeqClsModel(2,'/root/baselines/ctc_ner/macbert_correction/pycorrector-master/pycorrector/macbert/output/macbert4csc')
    model.load_state_dict(torch.load("model/model_pipeline.pt", map_location=torch.device('cpu')), strict=True)
    model.to("cuda:0")
    model.eval()
    model_checkpoint = '/root/baselines/ctc_ner/macbert_correction/pycorrector-master/pycorrector/macbert/output/macbert4csc'

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, model_max_length=128)
    all_class = []
    with torch.no_grad():
        for text in texts:
            t=tokenizer(text, truncation=True)
            logit=model(input_ids=torch.tensor([t['input_ids']]).to("cuda:0"),attention_mask=torch.tensor([t['attention_mask']]).to("cuda:0"))
            single_class=torch.argmax(logit['logits']).item()
            single_logit=torch.max(logit['logits']).item()
            all_class.append((single_class,single_logit))
    return all_class



