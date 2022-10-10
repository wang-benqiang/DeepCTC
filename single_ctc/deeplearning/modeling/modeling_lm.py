

import torch
from src.deeplearning.loss.focal_loss import FocalCELoss
from torch.cuda.amp import autocast
from transformers.models.bert import BertModel, BertPreTrainedModel


class ModelLmCscBert(BertPreTrainedModel):
    def __init__(self, config):
        super(ModelLmCscBert, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        
        self.sentence_detector = torch.nn.Linear(config.hidden_size, config.sentence_label_num)
        
        if config.add_detect_token:
            self.token_detector = torch.nn.Linear(config.hidden_size, config.token_label_num)

    @staticmethod
    def build_dummpy_inputs():
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(8, 56)))
        inputs['attention_mask'] = torch.ones(size=(8, 56)).long()
        inputs['position_ids'] = torch.zeros(size=(8, 56)).long()
        inputs['token_type_ids'] = torch.zeros(size=(8, 56)).long()
        return inputs
    
    
    def _init_criterion(self,):
        
        self.sentence_loss = FocalCELoss(loss_labels_weights=[1, 1.3])
        self.token_loss = FocalCELoss(loss_labels_weights=[1, 8])
    @autocast()
    def forward(self, 
                input_ids, 
                attention_mask, 
                sentence_detect_tags=None,
                token_detect_tags=None,
                pos_input_ids= None,
                pos_attention_mask= None,
                pos_sentence_detect_tags= None,
                pos_token_detect_tags= None
                ):
        
        if pos_input_ids is not None:
            # train with pos samples
            input_ids = torch.cat((input_ids, pos_input_ids), dim=0)
            attention_mask = torch.cat((attention_mask, pos_attention_mask), dim=0)
            sentence_detect_tags = torch.cat((sentence_detect_tags, pos_sentence_detect_tags), dim=0)
            token_detect_tags = torch.cat((token_detect_tags, pos_token_detect_tags), dim=0)
            
        
        
        h = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
      
        )[0]
        
        sentence_h = self.sentence_detector(h[:,0,:])
        
        token_h = self.token_detector(h)
        
        total_loss, token_loss, sent_loss =0, 0 ,0
        
        if token_detect_tags is not None and self.config.add_detect_token:
            
            token_loss = self.token_loss(token_h.view(-1, self.config.token_label_num), token_detect_tags.view(-1))
            
        if sentence_detect_tags is not None:
            sent_loss = self.sentence_loss(sentence_h.view(-1, self.config.sentence_label_num), sentence_detect_tags.view(-1))
            
        total_loss = token_loss + sent_loss
        
        return sentence_h, token_h, total_loss
    
    
    @staticmethod
    def build_dummpy_inputs():
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(torch.randint(low=1, high=10, size=(8, 56)))
        inputs['attention_mask'] = torch.ones(size=(8, 56)).long()
        inputs['sentence_labels'] = torch.ones(size=(8,)).long()
        inputs['token_labels'] = torch.ones(size=(8, 56)).long()
        return inputs

if __name__ == '__main__':
    
    model = ModelLmCscBert.from_pretrained('model/ctc_csc_no_punc_pretrain_2022Y06M18D18H/epoch3,ith_db0,step88600,testf1_63_94%,devf1_82_28%')
    model._init_criterion()
    inputs = model.build_dummpy_inputs()
    out = model(**inputs)
    print(out)
    
    