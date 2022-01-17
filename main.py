from numpy.lib.function_base import append
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import linear_model
from scipy.sparse import *
import numpy as np
import csv
import random
import copy
from train import PredictModel
csv.field_size_limit(500*1024*1024)

######### Extract source code #########
def code2str(cont):
    codestr = cont.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    codestr_t = codestr.split(' ')
    codestr = ''
    for i in codestr_t:
        if i != '':
            codestr += i + ' '
    return codestr

codecont=[]
with open("code_s.csv",'r') as f:
    cf=csv.reader(f)
    for line in cf:
        tmp=code2str(line[2]) #format  0:num 1:path 2:code
        codecont.append(tmp)
file_num=len(codecont)

######### Get code metrics #########
metrics=[]
filepath=[]
poses=[]
with open("metrics_s.csv", 'r') as f:
    cf=csv.reader(f)
    for line in cf:
        tmp=[]
        filepath.append(line[0]) #get file path
        for i in range(1,len(line)-1):
            tmp.append(eval(line[i]))
        metrics.append(tmp) #get file metrics
        t=len(filepath)-1
        if line[-1] == '1':
            poses.append(len(filepath)-1)
metrics=np.array(metrics)
print("poses ",poses)
print("metrics ",metrics.shape)
print("filepath ",len(filepath))



######### Calculate tf-idf score #########
def cal_tfidf(fea_num):
    global codecont
    tfidfer = TfidfVectorizer(lowercase=False,stop_words=None,norm=None, use_idf=True, smooth_idf=False,
                                sublinear_tf=False,decode_error="ignore")
    tfidf = tfidfer.fit_transform(codecont)
    weight = tfidf.sum(axis=0).tolist()[0]

    file_num=len(codecont)                              # files num

    kept = np.argsort(weight)[-fea_num:]
    voc = np.array(list(tfidfer.vocabulary_.keys()))[np.argsort(list(tfidfer.vocabulary_.values()))][kept]
    def checkvoc(voc):
        voc_str=''
        for i in range(len(voc)):
            voc_str += voc[i]
            voc_str += ','
        voc_str=voc_str.rstrip(',')
        with open("test/data1_word.txt" , "w") as f:
            f.write(voc_str)
        return voc_str
    #voc_str=checkvoc(voc)

    tfer = TfidfVectorizer(lowercase=False, stop_words=None,norm=u'l2', use_idf=False,
                            vocabulary=voc,decode_error="ignore")
    csr_mat=tfer.fit_transform(codecont)
    return csr_mat

fea_num=1500 #!#
csr_mat=cal_tfidf(fea_num)



######### Parameter set #########
def get_codelabel(poses,file_num):
    codelabel={}
    for i in range(file_num):
        if i in poses:
            codelabel[i]=1
        else:
            codelabel[i]=0
    return codelabel
codelabel=get_codelabel(poses,file_num)

predict_model=PredictModel()
predict_model.nL=copy.deepcopy(codelabel)
predict_model.prelabel=np.zeros([1,file_num],dtype=int)[0]
Trec=0.95 #!#
batch=10 # choosen file batch size #!#
ok=0

mode=0  #0:static 1:dynamic #!#
epoch=0

while len(predict_model.nL)>0 :
    vfiles={}                     
    
    if len(predict_model.Lr) < 1:
        #initial sampling
        vfiles=predict_model.Random()
    else:
        #sampling and train
        #predict_model.Random()
        predict_model.Train(csr_mat)
        vfiles=predict_model.Query(csr_mat,batch)
        ok=1
    #update each set
    
    print(vfiles)
    predict_model.Update(vfiles,poses)
    
    #break condition
    if ok!=0 and len(predict_model.Lr)>Trec*predict_model.Estimate(csr_mat,mode,poses):
        break
    epoch+=1
    vfiles.clear()
    print(vfiles)
        

print(epoch)
print(predict_model.Lr)
print(len(predict_model.Lr))
print(len(predict_model.L))
print(len(predict_model.prelabel))