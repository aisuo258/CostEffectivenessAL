import csv
csv.field_size_limit(500*1024*1024)

software_metrics_path="Mozilla_Firefox_Vulnerability_Data-master\software_metrics.csv"
vulnerabilities_path="Mozilla_Firefox_Vulnerability_Data-master/vulnerabilities.csv"
firefox_path="E:/Download/firefox/"


f1=open(software_metrics_path,"r")
#f2=open("code_s.csv","w",encoding = "utf8", newline="")
f2=open("code_CE.csv","w",encoding = "utf8", newline="")
cf1=csv.reader(f1, delimiter=',')
cf2=csv.writer(f2)
i=0
for line in cf1:
    #print(line[0])
    ### small
    if i>10000:
        break
    ###
    if i==0:
        i+=1
        continue
    cont=""
    with open(firefox_path+line[0],'r',encoding = "ISO-8859-1") as f3:
        tmps=f3.readlines()
        for tmp in tmps:
            tmp.strip("\"")
            cont+=tmp.replace("\n"," ")
    cf2.writerow([i,line[0],cont])
    i+=1
f1.close()
f2.close()

#####################
f1=open(software_metrics_path,"r",encoding = "ISO-8859-1")
f2=open(vulnerabilities_path,"r",encoding = "ISO-8859-1")
#f3=open("metrics_s.csv",'w',encoding = "utf8", newline="") 
f3=open("metrics_CE.csv",'w',encoding = "utf8", newline="") 
cf1=csv.reader(f1, delimiter=',')
cf2=csv.reader(f2, delimiter=',')
cf3=csv.writer(f3)
vulner=[]
for line in cf2:
    vulner.append(line[2])
vulner.pop(0)
i=0
for line in cf1:
    if i>10000:
        break
    i+=1
    if line[0] == "file":
        continue
    #print(line[0])
    if line[0] in vulner:
        ok=1
    else:
        ok=0
    tmp=line
    tmp.append(ok)
    cf3.writerow(tmp)

f1.close()
f2.close()
f3.close() 