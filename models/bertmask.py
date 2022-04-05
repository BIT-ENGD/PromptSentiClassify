
import torch.nn as nn 
from transformers import BertModel,BertForMaskedLM
class Bert_Model(nn.Module):
    def __init__(self,  bert_path ,config_file ):
        super(Bert_Model, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path,config=config_file)  # 加载预训练模型权重


    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids) #masked LM 输出的是 mask的值 对应的ids的概率 ，输出 会是词表大小，里面是概率 
        logit = outputs[0]  # 池化后的输出 [bs, config.hidden_size]


        return logit 
