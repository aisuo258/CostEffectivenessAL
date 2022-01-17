from numpy.lib.function_base import append
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import linear_model
from scipy.sparse import *
import numpy as np
import csv
import random
import copy
csv.field_size_limit(500*1024*1024)

class PredictModel():
    nL={}
    L={}
    Lr={}
    prelabel=np.zeros([1,1],dtype=int)[0]
    clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
    Ltrain={}

    def __init__(self):
        self.nL={}
        self.L={}
        self.Lr={}
        self.prelabel=np.zeros([1,1],dtype=int)[0]
        self.clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
        self.Ltrain={}
    
    def Random(self,ok):
        L_len=len(self.L)
        if L_len<=0:
            n=10    #初始随机个数 #!#
        else:
            n = L_len if len(self.nL)>L_len else len(self.nL)
        self.Ltrain={}
        a = random.sample(self.nL.keys(), n)
        for i in a:
            self.Ltrain[i]=0
        if ok!=0:
            for i in self.L: 
                self.Ltrain[i]=self.L[i]
        return self.Ltrain
    
    def Train(self,csr_mat):
        self.clf.fit(csr_mat,self.prelabel)
        nLk=list(self.nL.keys())
        if len(self.Lr)>10:
            Li=[]
            for i in self.Ltrain:
                if self.Ltrain[i]==0:
                    Li.append(i)
            Li+=nLk
            trans=self.clf.decision_function(csr_mat[Li])
            pos_at=list(self.clf.classes_).index(1)
            if pos_at:
                trans=-trans
            tmp_list=list(np.argsort(trans)[::-1][:len(self.Lr)])
            tmp=[]
            for i in tmp_list:
                tmp.append(Li[i])
            tmp+=list(self.Lr.keys())        
            self.clf = svm.SVC(kernel='linear', probability=True)
            self.clf.fit(csr_mat[tmp],self.prelabel[tmp])
        else:
            trans=self.clf.decision_function(csr_mat[nLk])
            pos_at=list(self.clf.classes_).index(1)
            if pos_at:
                trans=-trans
            tmp_list=list(np.argsort(trans)[::-1][:int(len(self.Lr)/2)])
            tmp=[]
            for i in tmp_list:
                tmp.append(nLk[i])
            tmp+=list(self.Ltrain.keys())
            self.clf.fit(csr_mat[tmp],self.prelabel[tmp])
        return self.clf    

    def Query(self,csr_mat,step):
        pos_at = list(self.clf.classes_).index(1)

        if len(self.nL)==0:
            return [],[]
        pool=list(self.nL.keys())
        if len(self.Lr)> step:#certain
            prob = self.clf.predict_proba(csr_mat[pool])[:,pos_at]
            order = np.argsort(prob)[::-1][:step]
        else:    #uncertain
            train_dist = self.clf.decision_function(csr_mat[pool])
            order = np.argsort(np.abs(train_dist))[:step]
        lsample={}
        for i in np.array(pool)[order]:
            lsample[i]=self.nL[i]
        #print("lsample ",len(lsample))
        return lsample
    
    def Update(self,vfiles,poses):
        for vfile in vfiles:
            if vfile in poses:
                self.L[vfile]=1
                self.Lr[vfile]=1
                self.prelabel[vfile]=1
            else:
                self.L[vfile]=0
            del self.nL[vfile]
    
    def Estimate(self,csr_mat,mode,Trec,poses,file_num):
        if mode==0:
            if len(self.Lr)<len(poses)*Trec:
                return 0
            else:
                return 1
        if mode ==1:
            if len(self.L)<0.45*file_num:
                return 0
            else:
                return 1
