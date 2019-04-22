import csv
csv_file=csv.reader(open('monthlstmDataSet.csv','r'))
Train_X=[]
Train_y=[]
Test_X=[]
Test_y=[]
n=0
for i in csv_file:
    if n<2500:
        count_t=0
        tmp=[]
        tmpAll=[]
        for j in range(2,38):
            if count_t<6:
                tmp.append(float(i[j]))
            else:
                tmpAll.append(tmp)
                tmp=[]
                tmp.append(float(i[j]))
                count_t=0
            count_t += 1
        tmpAll.append(tmp)
        Train_X.append(tmpAll)
        Train_y.append(i[-3:])
    elif n<3500:
        count_t = 0
        tmp = []
        tmpAll = []
        for j in range(2, 38):
            if count_t < 6:
                tmp.append(float(i[j]))
            else:
                tmpAll.append(tmp)
                tmp = []
                tmp.append(float(i[j]))
                count_t = 0
            count_t += 1
        Test_X.append(tmpAll)
        Test_y.append(i[-3:])
    n+=1

with open('TrainX.csv','w',newline='') as f:
    writer=csv.writer(f)
    for row in Train_X:
        writer.writerow(row)
    f.close()
with open('TrainY.csv','w',newline='') as f:
    writer=csv.writer(f)
    for row in Train_y:
        writer.writerow(row)
    f.close()

print(Train_X)
#print(len(Test_X))
