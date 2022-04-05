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

from models.bertmask import Bert_Model

from dataset.dataloader import load_data,MyDataSet,ProcessData,MASK_POS
# hyperparameters 
EPOCH=200
RANDOM_SEED=2022 
TRAIN_BATCH_SIZE=32  #小批训练， 批大小增大时需要提升学习率  https://zhuanlan.zhihu.com/p/413656738
TEST_BATCH_SIZE=96   #大批测试
EVAL_PERIOD=20
MODEL_NAME="bert-large-uncased"  # bert-base-chinese
DATA_PATH="data/Twitter2013"
NUM_WORKERS=10

train_file="twitter-2013train-A.tsv"
dev_file="twitter-2013dev-A.tsv"
test_file="twitter-2013test-A.tsv"


# env variables

os.environ['TOKENIZERS_PARALLELISM']="false" 

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter('./tb_log')
'''

'''

pd.options.display.max_columns = None
pd.options.display.max_rows = None








tokenizer=BertTokenizerFast.from_pretrained(MODEL_NAME)

config=BertConfig.from_pretrained(MODEL_NAME)

model=Bert_Model(bert_path=MODEL_NAME,config_file=config).to(device)



# get the data and label

# DATA_PATH+os.sep+filepath


Inputid_train,Labelid_train,typeids_train,inputnmask_train=ProcessData(DATA_PATH+os.sep+train_file,tokenizer)
Inputid_dev,Labelid_dev,typeids_dev,inputnmask_dev=ProcessData(DATA_PATH+os.sep+dev_file,tokenizer)
Inputid_test,Labelid_test,typeids_test,inputnmask_test=ProcessData(DATA_PATH+os.sep+test_file,tokenizer)


train_dataset = Data.DataLoader(MyDataSet(Inputid_train,  inputnmask_train , typeids_train , Labelid_train), TRAIN_BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
valid_dataset = Data.DataLoader(MyDataSet(Inputid_dev,  inputnmask_dev , typeids_dev , Labelid_dev), TRAIN_BATCH_SIZE,  shuffle=True,num_workers=NUM_WORKERS)
test_dataset = Data.DataLoader(MyDataSet(Inputid_test,  inputnmask_test , typeids_test , Labelid_test), TEST_BATCH_SIZE,  shuffle=True,num_workers=NUM_WORKERS)

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