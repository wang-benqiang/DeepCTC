import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import fastNLP
from fastNLP import Trainer
from fastNLP import Accuracy
import json
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# dataset = load_dataset('glue', 'sst2')
model_checkpoint = '../bert_dir/macbert4csc'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True,model_max_length=128)


lang8_data=json.load(open('lang8.json','r',encoding='utf-8'))

cged_data=json.load(open('cged.json','r',encoding='utf-8'))

lang8_text,lang8_label=lang8_data[0],lang8_data[1]
cged_text,cged_label=cged_data[0],cged_data[1]


lang8_dataset=[]
for text,label in zip(lang8_text,lang8_label):
    single_sample=tokenizer(text, truncation=True)
    single_sample['label']=label
    lang8_dataset.append(single_sample)

random.shuffle(lang8_dataset)
lang8_train_dataset,lang8_dev_dataset=lang8_dataset[10000:],lang8_dataset[:10000]

cged_dataset=[]
for text,label in zip(cged_text,cged_label):
    single_sample=tokenizer(text, truncation=True)
    single_sample['label']=label
    cged_dataset.append(single_sample)

random.shuffle(cged_dataset)
cged_train_dataset,cged_dev_dataset=cged_dataset[5000:],cged_dataset[:5000]



class SeqClsDataset(Dataset):
    def __init__(self, dataset):
        Dataset.__init__(self)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        item = self.dataset[item]
        return item['input_ids'], item['attention_mask'], [item['label']]

def collate_fn(batch):
    input_ids, atten_mask, labels = [], [], []
    max_length = [0] * 3
    for each_item in batch:
        input_ids.append(each_item[0])
        max_length[0] = max(max_length[0], len(each_item[0]))
        atten_mask.append(each_item[1])
        max_length[1] = max(max_length[1], len(each_item[1]))
        labels.append(each_item[2])
        max_length[2] = max(max_length[2], len(each_item[2]))

    for i in range(3):
        each = (input_ids, atten_mask, labels)[i]
        for item in each:
            item.extend([0] * (max_length[i] - len(item)))
    return {'input_ids': torch.cat([torch.tensor([item]) for item in input_ids], dim=0),
            'attention_mask': torch.cat([torch.tensor([item]) for item in atten_mask], dim=0),
            'labels': torch.cat([torch.tensor(item) for item in labels], dim=0)}

lang8_dataset_train = SeqClsDataset(lang8_train_dataset)
lang8_dataloader_train = DataLoader(dataset=lang8_dataset_train,
                              batch_size=128, shuffle=True, collate_fn=collate_fn)
lang8_dataset_valid = SeqClsDataset(lang8_dev_dataset)
lang8_dataloader_valid = DataLoader(dataset=lang8_dataset_valid,
                              batch_size=16, shuffle=False, collate_fn=collate_fn)

cged_dataset_train = SeqClsDataset(cged_train_dataset)
cged_dataloader_train = DataLoader(dataset=cged_dataset_train,
                              batch_size=128, shuffle=True, collate_fn=collate_fn)
cged_dataset_valid = SeqClsDataset(cged_dev_dataset)
cged_dataloader_valid = DataLoader(dataset=cged_dataset_valid,
                              batch_size=16, shuffle=False, collate_fn=collate_fn)

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

num_labels =2

model = SeqClsModel(num_labels=num_labels, model_checkpoint=model_checkpoint)

optimizers = AdamW(params=model.parameters(), lr=5e-5)

lang8_trainer = Trainer(
    model=model,
    driver='torch',
    device=[3,4,5,6],  # 'cuda'
    n_epochs=15,
    optimizers=optimizers,
    fp16=True,
    train_dataloader=lang8_dataloader_train,
    evaluate_dataloaders=lang8_dataloader_valid,
    metrics={'acc': Accuracy()}
)

lang8_trainer.run(num_eval_batch_per_dl=10)

cged_trainer = Trainer(
    model=model,
    driver='torch',
    device=[3,4,5,6],  # 'cuda'
    n_epochs=15,
    optimizers=optimizers,
    train_dataloader=cged_dataloader_train,
    evaluate_dataloaders=cged_dataloader_valid,
    metrics={'acc': Accuracy()}
)

cged_trainer.run(num_eval_batch_per_dl=10)

model_to_save = (model.module if hasattr(model, "module") else model)
torch.save(model_to_save.state_dict(), 'model_pipeline.pt')
