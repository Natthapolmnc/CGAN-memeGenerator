import requests as req
import pandas as pd
import pickle as pkl
import os
from tqdm import tqdm
import numpy as np
import time
from multiprocessing import Pool


ori_dat=pd.read_csv("raw_dat\memes_reference_data.tsv",sep="\t")
meme_dat=pd.read_csv("raw_dat\memes_data.tsv",sep="\t")

merge_df=ori_dat.merge(meme_dat,how="outer",on="MemeLabel")


df=merge_df.drop(columns=["StandardTextBox","HashId"])
df["CaptionLen"]=df["CaptionText"].apply(lambda x:len(x))
df=df.where(df["CaptionLen"]>=10).dropna()
df=df.drop_duplicates()


msk = np.random.rand(len(df)) < 0.8
train=df[msk]
test=df[~msk]

baseimg_id_dict={v:k for k,v in enumerate(ori_dat["MemeLabel"].unique())}
img_id_dict={v:k for k,v in enumerate(df["ImageURL"])}
id_oriname_dic={baseimg_id_dict[row["MemeLabel"]]:"pre_dat/ori_pic/"+str(baseimg_id_dict[row["MemeLabel"]])+"_ori.jpg" for index,row in ori_dat.iterrows()}
id_memetrn_dic={img_id_dict[row["ImageURL"]]:"pre_dat/meme_pic_train/"+str(img_id_dict[row["ImageURL"]])+"tr_img.jpg" for index,row in train.iterrows()}
id_memetst_dic={img_id_dict[row["ImageURL"]]:"pre_dat/meme_pic_test/"+str(img_id_dict[row["ImageURL"]])+"ts_img.jpg" for index,row in test.iterrows()}


chunks_train=np.array_split(train, 3000)
chunks_test=np.array_split(test, 1000)

for index,row in ori_dat.iterrows():
    res=req.get(row["BaseImageURL"])
    file_ori=open("pre_dat/ori_pic/"+str(baseimg_id_dict[row["MemeLabel"]])+"_ori.jpg","wb")
    id_oriname_dic[baseimg_id_dict[row["MemeLabel"]]]="pre_dat/ori_pic/"+str(baseimg_id_dict[row["MemeLabel"]])+"_ori.jpg"
    file_ori.write(res.content)
    file_ori.close()

def dwnlMemeTrain(df):
    for index,row in df.iterrows():
        try:
            res=req.get("http:"+row["ImageURL"])
            file_meme=open("pre_dat/meme_pic_train/"+str(img_id_dict[row["ImageURL"]])+"trn_img.jpg","wb")
            id_memetrn_dic[img_id_dict[row["ImageURL"]]]="pre_dat/meme_pic_train/"+str(baseimg_id_dict[row["MemeLabel"]])+"tr_img.jpg"
            file_meme.write(res.content)
            file_meme.close()
        except:
            time.sleep(10)
            res=req.get("http:"+row["ImageURL"])
            file_meme=open("pre_dat/meme_pic_train/"+str(img_id_dict[row["ImageURL"]])+"trn_img.jpg","wb")
            id_memetrn_dic[img_id_dict[row["ImageURL"]]]="pre_dat/meme_pic_train/"+str(baseimg_id_dict[row["MemeLabel"]])+"tr_img.jpg"
            file_meme.write(res.content)
            file_meme.close()
            
def dwnlMemeTest(df):
    for index,row in df.iterrows():
        try:
            res=req.get("http:"+row["ImageURL"])
            file_meme=open("pre_dat/meme_pic_test/"+str(img_id_dict[row["ImageURL"]])+"ts_img.jpg","wb")
            id_memetst_dic[img_id_dict[row["ImageURL"]]]="pre_dat/meme_pic_test/"+str(baseimg_id_dict[row["MemeLabel"]])+"ts_img.jpg"
            file_meme.write(res.content)
            file_meme.close()
        except:
            time.sleep(10)
            res=req.get("http:"+row["ImageURL"])
            file_meme=open("pre_dat/meme_pic_test/"+str(img_id_dict[row["ImageURL"]])+"ts_img.jpg","wb")
            id_memetst_dic[img_id_dict[row["ImageURL"]]]="pre_dat/meme_pic_test/"+str(baseimg_id_dict[row["MemeLabel"]])+"ts_img.jpg"
            file_meme.write(res.content)
            file_meme.close()

if __name__ == "__main__":
    try:
        pool = Pool(processes=7)
        result=list(tqdm(pool.imap(dwnlMemeTrain,chunks_train),total=len(chunks_train),desc="train image downloading...."))
        pool.close()
        pool.join()
        pool = Pool(processes=7)
        result=list(tqdm(pool.imap(dwnlMemeTest,chunks_test),total=len(chunks_test),desc="test image downloading..."))
        pool.close()
        pool.join()

        train["BaseImg"]=train["MemeLabel"].apply(lambda x:id_oriname_dic[baseimg_id_dict[x]])
        test["BaseImg"]=test["MemeLabel"].apply(lambda x:id_oriname_dic[baseimg_id_dict[x]])
        train["MemeImg"]=train["ImageURL"].apply(lambda x:id_memetrn_dic[img_id_dict[x]])
        test["MemeImg"]=test["ImageURL"].apply(lambda x:id_memetst_dic[img_id_dict[x]])

        train.to_csv("train_df.csv")
        test.to_csv("test_df.csv")
    except:
        print ("ERROR I KWAI")