#构建数据集
import torch.utils.data as Data
import pandas as pd
import torch 

PREFIX = 'It was [mask]. '
MASK_POS=3  # "it was [mask]" 中 [mask] 位置





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


def ProcessData(filepath,tokenizer):
    pos_id=tokenizer.convert_tokens_to_ids("good") #9005
    neg_id=tokenizer.convert_tokens_to_ids("bad")  #12139
    x_train,y_train=load_data(filepath)
    #x_train,x_test,y_train,y_test=train_test_split(StrongData,StrongLabel,test_size=0.3, random_state=42)

    Inputid=[]
    Labelid=[]
    typeid=[]
    attenmask=[]

    for i in range(len(x_train)):

        text_ = PREFIX+x_train[i][0]

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
