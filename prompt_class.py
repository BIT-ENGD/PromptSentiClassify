'''

https://blog.csdn.net/wf19971210/article/details/120543015

链接：https://pan.baidu.com/s/1Nx7htUBWKBZfo3QPPty3mw
提取码：1234
'''
import warnings
from datetime import datetime
import time 
import torch
import os
from transformers import BertModel,BertConfig,BertModel,BertTokenizerFast,get_cosine_schedule_with_warmup,BertForMaskedLM
import pandas  as pd
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter 

# hyperparameters 
EPOCH=200
RANDOM_SEED=2022 
TRAIN_BATCH_SIZE=32  #小批训练， 批大小增大时需要提升学习率  https://zhuanlan.zhihu.com/p/413656738
TEST_BATCH_SIZE=96   #大批测试
EVAL_PERIOD=20
MODEL_NAME="bert-large-uncased"  # bert-base-chinese
DATA_PATH="data/Twitter2013"
MASK_POS=3  # "it was [mask]" 中 [mask] 位置
train_file="twitter-2013train-A.tsv"
dev_file="twitter-2013dev-A.tsv"
test_file="twitter-2013test-A.tsv"

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter('./tb_log')
'''

'''

pd.options.display.max_columns = None
pd.options.display.max_rows = None




prefix = 'It was [mask]. '

class Bert_Model(nn.Module):
    def __init__(self,  bert_path ,config_file ):
        super(Bert_Model, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_path,config=config_file)  # 加载预训练模型权重


    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids) #masked LM 输出的是 mask的值 对应的ids的概率 ，输出 会是词表大小，里面是概率 
        logit = outputs[0]  # 池化后的输出 [bs, config.hidden_size]


        return logit 


#构建数据集
class MyDataSet(Data.Dataset):
    def __init__(self, sen , mask , typ ,label ):
        super(MyDataSet, self).__init__()
        self.sen = torch.tensor(sen,dtype=torch.long)
        self.mask = torch.tensor(mask,dtype=torch.long)
        self.typ =torch.tensor( typ,dtype=torch.long)
        self.label = torch.tensor(label,dtype=torch.long)

    def __len__(self):
        return self.sen.shape[0]

    def __getitem__(self, idx):
        return self.sen[idx], self.mask[idx],self.typ[idx],self.label[idx]
#load  data
   
def load_data(tsvpath):
    data=pd.read_csv(tsvpath,sep="\t",header=None,names=["sn","polarity","text"])
    data=data[data["polarity"] != "neutral"]
    yy=data["polarity"].replace({"negative":0,"positive":1,"neutral":2})
    # print(data.loc[0:5,[0,1]])  # 
    #print(data.iloc[0:5,[1,1]])  # 
    #print(data.iloc[:,1:2])  # 
    #print(data.iloc[:,2:3])  # 
    return data.values[:,2:3].tolist(),yy.tolist() #data.values[:,1:2].tolist()

tokenizer=BertTokenizerFast.from_pretrained(MODEL_NAME)

config=BertConfig.from_pretrained(MODEL_NAME)

model=Bert_Model(bert_path=MODEL_NAME,config_file=config).to(device)

pos_id=tokenizer.convert_tokens_to_ids("good") #9005
neg_id=tokenizer.convert_tokens_to_ids("bad")  #12139



# get the data and label

def ProcessData(filepath):
    x_train,y_train=load_data(DATA_PATH+os.sep+filepath)
    #x_train,x_test,y_train,y_test=train_test_split(StrongData,StrongLabel,test_size=0.3, random_state=42)

    Inputid=[]
    Labelid=[]
    typeid=[]
    attenmask=[]

    for i in range(len(x_train)):

        text_ = prefix+x_train[i][0]

        encode_dict = tokenizer.encode_plus(text_,max_length=60,padding="max_length",truncation=True)
        input_ids=encode_dict["input_ids"]
        type_ids=encode_dict["token_type_ids"]
        atten_mask=encode_dict["attention_mask"]
        labelid,inputid= input_ids[:],input_ids[:]
        if y_train[i] == 0:
            labelid[MASK_POS] = neg_id
            labelid[:MASK_POS] = [-1]* len(labelid[:MASK_POS]) 
            labelid[MASK_POS+1:] = [-1] * len(labelid[MASK_POS+1:])
            inputid[MASK_POS] = tokenizer.mask_token_id
        else:
            labelid[MASK_POS] = pos_id
            labelid[:MASK_POS] = [-1]* len(labelid[:MASK_POS]) 
            labelid[MASK_POS+1:] = [-1] * len(labelid[MASK_POS+1:])
            inputid[MASK_POS] = tokenizer.mask_token_id

        Labelid.append(labelid)
        Inputid.append(inputid)
        typeid.append(type_ids)
        attenmask.append(atten_mask)

    return Inputid,Labelid,typeid,attenmask



Inputid_train,Labelid_train,typeids_train,inputnmask_train=ProcessData(train_file)
Inputid_dev,Labelid_dev,typeids_dev,inputnmask_dev=ProcessData(dev_file)
Inputid_test,Labelid_test,typeids_test,inputnmask_test=ProcessData(test_file)


train_dataset = Data.DataLoader(MyDataSet(Inputid_train,  inputnmask_train , typeids_train , Labelid_train), TRAIN_BATCH_SIZE, True)
valid_dataset = Data.DataLoader(MyDataSet(Inputid_dev,  inputnmask_dev , typeids_dev , Labelid_dev), TRAIN_BATCH_SIZE, True)
test_dataset = Data.DataLoader(MyDataSet(Inputid_test,  inputnmask_test , typeids_test , Labelid_test), TEST_BATCH_SIZE, True)

train_data_num=len(Inputid_train)
test_data_num=len(Inputid_test)
#print("hello!")



optimizer = optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4)  #使用Adam优化器
loss_func = nn.CrossEntropyLoss(ignore_index=-1)
EPOCH = 200
schedule = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=len(train_dataset),num_training_steps=EPOCH*len(train_dataset))
print("正在训练中。。。")
totaltime=0
for epoch in range(EPOCH):

    starttime_train=datetime.now()

    start =time.time()
    correct=0
    train_loss_sum=0
    model.train()

    for idx,(ids,att_mask,type,y) in enumerate(train_dataset):
        ids,att_mask,type,y = ids.to(device),att_mask.to(device),type.to(device),y.to(device)
        out_train = model(ids,att_mask,type)
       #print(out_train.view(-1, tokenizer.vocab_size).shape, y.view(-1).shape)
        loss = loss_func(out_train.view(-1, tokenizer.vocab_size),y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        train_loss_sum += loss.item()
       
        if( idx+1)% EVAL_PERIOD == 0:
            print("Epoch {:04d} | Step {:06d}/{:06d} | Loss {:.4f} | Time {:.0f}".format(
                epoch + 1, idx + 1, len(train_dataset), train_loss_sum / (idx + 1), time.time() - start))
            writer.add_scalar('loss/train_loss', train_loss_sum / (idx + 1), epoch)

        truelabel=y[:,MASK_POS]
        out_train_mask=out_train[:,MASK_POS,:]
        predicted=torch.max(out_train_mask,1)[1]
        correct += (predicted == truelabel).sum()
        correct = float(correct)
    
    acc =float(correct /train_data_num)

    eval_loss_sum=0.0
    model.eval()
    correct_test=0
    with torch.no_grad():
        for ids, att, tpe, y in test_dataset:
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            out_test = model(ids , att , tpe)
            loss_eval = loss_func(out_test.view(-1, tokenizer.vocab_size), y.view(-1))
            eval_loss_sum += loss_eval.item()
            ttruelabel = y[:, MASK_POS]
            tout_train_mask = out_test[:, MASK_POS, :]
            predicted_test = torch.max(tout_train_mask.data, 1)[1]
            correct_test += (predicted_test == ttruelabel).sum()
            correct_test = float(correct_test)
    acc_test = float(correct_test / test_data_num)

    if epoch % 1 == 0:
        out = ("epoch {}, train_loss {},  train_acc {} , eval_loss {} ,acc_test {}"
               .format(epoch + 1, train_loss_sum / (len(train_dataset)), acc, eval_loss_sum / (len(test_dataset)),
                acc_test))
        writer.add_scalar('loss/test_loss', train_loss_sum / (idx + 1), epoch)
        print(out)
    end=time.time()

    print("epoch {} duration:".format(epoch+1),end-start)
    totaltime+=end-start

print("total training time: ",totaltime)
