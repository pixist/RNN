# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:59:42 2022

@author: novar
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

class RNNLayer:
    def forward(self, x, oldS, U, W, V):
        self.mulUX = np.dot(U, x)
        self.mulWos = np.dot(W, oldS)
        self.add = self.mulWos + self.mulUX
        self.s = np.tanh(self.add) #activation, s = hidden
        self.mulV = np.dot(V, self.s) #out
        
    def backward(self, x, oldS, U, W, V, sDiff, dmulv):
        #self.forward(x, oldS, U, W, V)
        dV = np.asarray(np.dot(np.asmatrix(dmulv).T, np.asmatrix(self.s))) #write this better, shorter
        dSv = np.dot(V.T,dmulv)
        ds = dSv + sDiff
        dadd = (1-np.square(np.tanh(self.add)))*ds
        dmulUX = dadd * np.ones_like(self.mulUX)
        dmulWos = dadd * np.ones_like(self.mulWos)
        dW = np.asarray(np.dot(np.asmatrix(dmulWos).T, np.asmatrix(oldS)))
        doldS = np.dot(W.T,dmulWos)
        dU = np.asarray(np.dot(np.asmatrix(dmulUX).T, np.asmatrix(x)))
        #dx = np.dot(U.T,dmulUX)
        return doldS, dU, dW, dV
# DONE


class RNN:
    def __init__(self, Lfeature, Lxdim, Lhid, Lclass, bptt):
        self.Lhid = Lhid
        self.Lfeature = Lfeature
        self.Lxdim = Lxdim
        self.Lclass = Lclass
        self.bptt = bptt
        
    def initializeWeights(self):
        self.U = np.random.uniform(-np.sqrt(1./self.Lxdim),np.sqrt(1./self.Lxdim),(self.Lhid,self.Lxdim))#weights between input and hidden layers
        self.W = np.random.uniform(-np.sqrt(1./self.Lhid),np.sqrt(1./self.Lhid),(self.Lhid,self.Lhid))
        self.V = np.random.uniform(-np.sqrt(1./self.Lhid),np.sqrt(1./self.Lhid),(self.Lclass,self.Lhid))
        self.dU = np.zeros(self.U.shape)
        self.dW = np.zeros(self.W.shape)
        self.dV = np.zeros(self.V.shape)
        self.dUnew = np.zeros(self.U.shape)
        self.dWnew = np.zeros(self.W.shape)
        self.dVnew = np.zeros(self.V.shape)
    
    def forward(self, data):
        #data is (T,) size timeseries, parellelize after implementing single steps
        T = len(data)
        foldedLayers = []
        oldS = np.zeros(self.Lhid)
        for t in range(T):
                rnnlayer = RNNLayer()
                rnnlayer.forward(data[t], oldS, self.U, self.W, self.V)
                oldS = rnnlayer.s
                foldedLayers.append(rnnlayer)
        self.rnnLayers = foldedLayers
        return foldedLayers
    
    def forwardOut(self):
        # call forward before to update self.rnnLayers
        self.out = np.zeros(self.Lclass)
        self.outProb = softmax(self.rnnLayers[-1].mulV)
        self.out[np.argmax(self.outProb)] = 1
        return self.out, self.outProb
    
    def crossEntropy(self, ground):
        # call forwardOut before to update self.out
        # do not forget to parallelize this for N samples
        return -np.sum(ground*np.log(self.outProb))
    
    def calcGrad(self, data, ground):
        #run after forward
        lyr = self.rnnLayers
        oldSt = np.zeros(self.Lhid)
        sDiff = np.zeros(self.Lhid)
        
        t = self.Lfeature - 1
        dmulV = self.outProb - ground
        doldS, dUt, dWt, dVt = lyr[t].backward(data[t], oldSt, self.U, self.W, self.V, sDiff, dmulV)
        oldSt = lyr[t].s
        dmulV = np.zeros(self.Lclass)
        for i in range(t-1, max(-1, t-self.bptt-1), -1): # no need for this much eleboration
            oldSi = np.zeros(self.Lhid) if i == 0 else lyr[i-1].s # remove this
            doldS, dUi, dWi, dVi = lyr[i].backward(data[i], oldSi, self.U, self.W, self.V, doldS, dmulV)
            dUt += dUi
            dWt += dWi
        self.dUnew = self.dUnew + dUt
        self.dWnew = self.dWnew + dWt
        self.dVnew = self.dVnew + dVt
        
    def updateWeights(self, learningRate, momentum):
        #run after grad calculation is done for enough samples depending on the batch size
        self.dU = momentum*self.dU + self.dUnew
        self.dW = momentum*self.dW + self.dWnew
        self.dV = momentum*self.dV + self.dVnew
        self.U = self.U - learningRate*self.dU
        self.W = self.W - learningRate*self.dW
        self.V = self.V - learningRate*self.dV
        self.dUnew = np.zeros(self.U.shape)
        self.dWnew = np.zeros(self.W.shape)
        self.dVnew = np.zeros(self.V.shape)

    def trainStep(self, sample, target):
        # calls required methods sequentally except updateWeights()
        self.forward(sample)
        guess, _ = self.forwardOut()
        loss = self.crossEntropy(target.reshape(target.shape[1]))
        self.calcGrad(sample, target.reshape(target.shape[1]))
        return loss, guess

def trainMiniBatch(nnModel, data, ground, valX, valD, testX, testD, epoch, learningRate, momentum, batchSize = 32):
    countSamples = 0
    lossListT, lossListV, accuracyListT, accTest= [], [], [], []
    totalSamples = len(ground)
    batchCount = totalSamples//batchSize
    remainder = totalSamples % batchSize
    remLimit = totalSamples - remainder
    for e in range(epoch):
        permutation = list(np.random.permutation(totalSamples))
        shuffled_samples = data[permutation]
        shuffled_grounds = ground[permutation]
        samples = np.array_split(shuffled_samples[:remLimit], batchCount)
        grounds = np.array_split(shuffled_grounds[:remLimit], batchCount)
        samples.append(shuffled_samples[remLimit:])
        grounds.append(shuffled_grounds[remLimit:])
        
        estimatesT = []
        loss = 0
        for j in range(len(grounds)):
            bSize = grounds[j].shape[0]
            for i in range(bSize):
                countSamples += 1
                l, g = nnModel.trainStep(samples[j][i], grounds[j][i][None,:])
                estimatesT.append(g)
                loss += l
            nnModel.updateWeights(learningRate, momentum)
        loss = loss/totalSamples
        lossListT.append(loss)
        
        gndidx = np.array([np.where(r==1)[0][0] for r in shuffled_grounds]) + 1
        estidx = np.array([np.where(r==1)[0][0] for r in estimatesT]) + 1
        
        falses = np.count_nonzero(gndidx-estidx)
        accuracy = 1-falses/totalSamples
        accuracyListT.append(accuracy)
        
        loss = 0
        for i in range(valD.shape[0]):
            nnModel.forward(valX[i])
            guess, _ = nnModel.forwardOut()
            loss += nnModel.crossEntropy(valD[i][None,:])
        loss = loss/valD.shape[0]
        lossListV.append(loss)
        
        estTest = []
        for i in range(testD.shape[0]):
            nnModel.forward(testX[i])
            guess, _ = nnModel.forwardOut()
            estTest.append(guess)
        
        Tgndidx = np.array([np.where(r==1)[0][0] for r in testD]) + 1
        estTestidx = np.array([np.where(r==1)[0][0] for r in estTest]) + 1
        
        falses = np.count_nonzero(Tgndidx-estTestidx)
        accuracy = 1-falses/testD.shape[0]
        accTest.append(accuracy)
        
        print(f"Validation Loss in epoch {e+1}: {loss}, Test Accuracy: {accuracy}")
        if loss > 1.2*lossListV[0]: 
            print("Termnated due to increased loss")
            return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)
        elif (e > 1) & (lossListT[e-1] - lossListT[e] < 0.0001):
            print("Terminated due to convergence")
            return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)
    return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def comp_confmat(actual, predicted):
    np.seterr(divide='ignore')
    classes = np.unique(actual)
    confmat = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))
    return confmat 

def plotTwinParameter(metric, labels):
    xlabel = [i for i in range(len(metric[0]))]
    plt.plot(xlabel, metric[0], marker='o', markersize=6, linewidth=2, label=labels[0])
    plt.legend()
    plt.ylabel(labels[0])
    plt.ylim((0,1.1))
    ax2 = plt.twinx()
    ax2.plot(xlabel, metric[1], marker='o', color = 'red', markersize=6, linewidth=2, label=labels[1])
    plt.ylabel(labels[1])
    plt.title('Parameter vs Metrics Plot')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
def plotParameter(metric, labels, metricName):
    plt.figure(figsize = (12,6))
    xlabel = [str(i) for i in range(len(metric[0]))]
    for i in range(len(labels)):
        plt.plot(xlabel, metric[i], marker='o', markersize=6, linewidth=2, label=labels[i])
    plt.ylabel(metricName[0])
    plt.title(f'{metricName[1]} with Learning Rate: {metricName[2]}, Momentum: {metricName[3]}, BPTT: {metricName[4]}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
def plotConf(mat_con, Title):
    fig, px = plt.subplots(figsize=(7.5, 7.5))
    px.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
    for m in range(mat_con.shape[0]):
        for n in range(mat_con.shape[1]):
            px.text(x=m,y=n,s=int(mat_con[m, n]), va='center', ha='center', size='xx-large')
    
    # Sets the labels
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title('Confusion Matrix for '+Title, fontsize=15)
    plt.show()

# In[Read the data]
filename = "data3.h5"

with h5py.File(filename, "r") as f:
    groupKeys = list(f.keys())
    sets = []
    for key in groupKeys:
        sets.append(list(f[key]))
del key
# In[]
idx = np.random.permutation(3000)
trainX = np.array(sets[0])[idx]
trainD = np.array(sets[1])[idx]
testX = np.array(sets[2])
testD = np.array(sets[3])
valX = trainX[:300]
valD = trainD[:300]
trainX = trainX[300:]
trainD = trainD[300:]
# In[]
bptt = 10
model = RNN(150, 3, 128, 6, bptt)
model.initializeWeights()
lossT, lossV, accT, accTest = [], [], [], []
# In[]
lr = 0.001
mm = 0.5
epoch = 10
print(f"Started Training with learning rate = {lr}, momentum = {mm}, bptt = {bptt}")
l1, l2, a1, a2, confT, confTest = trainMiniBatch(model, trainX, trainD, valX, valD, testX, testD, epoch, lr, mm)
lossT.extend(l1)
lossV.extend(l2) 
accT.extend(a1)
accTest.extend(a2)
# In[]
plotConf(confT, "Training Set, RNN")
plotConf(confTest, "Test Set, RNN")
# In[plot]
plotParameter([lossT, lossV], ["Training","Validation"], ["Loss","RNN",lr,mm,bptt])
#%%
plotParameter([accT, accTest], ["Training","Validation"], ["Accuracy","RNN",lr,mm,bptt])