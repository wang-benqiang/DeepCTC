#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertForMaskedLM
from src.loss.focal_loss import FocalCELoss
from torch.cuda.amp import autocast




