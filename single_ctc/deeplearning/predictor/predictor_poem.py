# coding: UTF-8
import json
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig


# %%
class PoemModel(nn.Module):

    def __init__(self, model_path, num_classes=2):
        super(PoemModel, self).__init__()
        self.hidden_size = 768
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrain_config = BertConfig.from_pretrained(model_path)
        self.bert = BertModel(self.pretrain_config)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                              return_dict=False)
        out = self.fc(pooled)
        return out


class PoemPredictor:

    def __init__(self, model_path):
        self.model_path = model_path
        self._config = self._load_config()
        self.model, self.tokenizer, self.device = self._load_model()

    def _load_config(self):
        with open('{}/train_config.json'.format(self.model_path), 'r', encoding="utf-8") as f:
            _config = json.load(f)
        return _config

    def _load_model(self):
        model = PoemModel(self.model_path)
        the_device = model.device
        the_model = torch.load(os.path.join(self.model_path, self._config["model_name"]), map_location="cpu")
        model.load_state_dict(the_model)
        tokenizer = model.tokenizer
        model.to(the_device)
        model.eval()
        return model, tokenizer, the_device

    def predict(self, text):
        tk_result = self.tokenizer(text)
        input_ids, attention_mask, token_type_ids = torch.unsqueeze(torch.LongTensor(tk_result["input_ids"]),
                                                                    dim=0).to(self.device), \
                                                    torch.unsqueeze(torch.LongTensor(tk_result["attention_mask"])
                                                                    , dim=0).to(self.device), \
                                                    torch.unsqueeze(torch.LongTensor(tk_result["token_type_ids"]),
                                                                    dim=0).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids, attention_mask, token_type_ids)
            label = int(torch.argmax(output).detach())
            label_out = self._config["tag_dict"]["id2tag"][str(label)]
        return label_out, output


