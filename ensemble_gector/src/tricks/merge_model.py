
import os
import torch
from src.modeling.modeling_csc_seq2edit import ModelingCscSeq2Edit
from src.modeling.modeling_csc_realise import ModelingCscRealise
from src.tokenizer.bert_tokenizer import CustomBertTokenizer


@torch.no_grad()
def merge_models(model_class, tokenizer_class, in_models_dir: str or list, out_model_dir: str):
    """_summary_

    Args:
        model_class (_type_): huggingface的model类
        in_models_dir (strorlist): _description_
        out_model_dir (str): _description_
    """

    if isinstance(in_models_dir, str):
        in_model_dir_list = [os.path.join(
            in_models_dir, model_dir_name) for model_dir_name in os.listdir(in_models_dir)]
    else:
        in_model_dir_list = in_models_dir
    model_num = len(in_model_dir_list)

    base_model = model_class.from_pretrained(in_model_dir_list[0])
    tokenizer = tokenizer_class.from_pretrained(in_model_dir_list[0])
    base_model.eval()
    for in_model_dir in in_model_dir_list[1:]:
        model = model_class.from_pretrained(in_model_dir)
        model.eval()
        for (base_model_param_name, base_model_param), (param_name, param) in zip(base_model.named_parameters(), model.named_parameters()):
            assert base_model_param_name == param_name
            if param.requires_grad:
                base_model_param.data += param.data

    for param_name, param in base_model.named_parameters():
        if param.requires_grad:
            param.data = param.data/model_num

    # save merged model
    base_model.save_pretrained(out_model_dir)
    tokenizer.save_pretrained(out_model_dir)


if __name__ == '__main__':
    models_dir = 'model/csc_realise_ccl_test_ft1_2022Y08M11D18H'
    merge_models(ModelingCscRealise, CustomBertTokenizer, models_dir,
                 out_model_dir='model/csc_realise_ccl_test_ft1_merge')
