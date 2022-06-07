import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys, os
import warnings
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
from sigfig import round as siground
import math



class Net(nn.Module):
    def __init__(self, *args):
        super(Net, self).__init__()
        sizes, self.convs, lins, csizes,  dropout, self.resnet, self.activation = args  
        
        if self.resnet:
            pads = ((np.array(csizes)-1)/2).astype(int)
            lsize = self.convs[-1] * sizes[0]*sizes[1]
        elif self.convs:
            pads = np.zeros(len(csizes)).astype(int)
            lsize = self.convs[-1]
            for i in range(len(sizes)):
                lsize *= (sizes[i] - sum(csizes) + len(csizes))
        else: 
            lsize = 1
            for i in range(len(sizes)):
                lsize *= sizes[i]
        
        ## Create List of Convolution Layers
        if len(self.convs):
            self.convs = np.append([1,], self.convs)
            self.cfcs = nn.ModuleList()        
            for i in range(len(self.convs)-1):
                self.cfcs.append(nn.Conv2d(self.convs[i], self.convs[i+1], csizes[i], padding = pads[i]).double())

        ## Create List of Linear Layers
        self.lfcs = nn.ModuleList()
        if not lins[-1]==1:
            lins.append(1)
        if lins:
            lins = np.append([lsize,], lins)
            for i in range(len(lins)-1):
                self.lfcs.append( nn.Linear(lins[i], lins[i+1], ))

        ## create a list of Activations
        self.acts = nn.ModuleList()
        for i in range(len(lins)):
            self.acts.append(self.activation())
        
        if dropout:
            self.drops = nn.ModuleList()
            for i in range(len(lins)):
                self.drops.append(nn.Dropout(dropout))
        else: self.drops = []
        

    def forward(self, output):   
        if len(self.convs):
            repnum = self.convs[0]
            output = output/torch.mean(output)
            for i in range(len(self.cfcs)):
                if self.resnet:
                    old  = output.repeat(1, repnum, 1, 1) 
                    new = self.cfcs[i](output)
                    output = new/torch.mean(new) + old
                    repnum = int((self.convs[(i+2)%len(self.convs)])/self.convs[i+1])
                else:
                    output = self.cfcs[i](output)
                     
        #Create Embedding  
        output = output.view(output.size()[0], -1)
        output = torch.unsqueeze(output, 1)
                             
        #Run linear layers and activations
        for i in range(len(self.lfcs)):
            output = self.lfcs[i](output)
            output = self.acts[i](output)
            if self.drops:
                output = self.drops[i](output)

        return output

    
class Model:
    def __init__(self, *args, **kwargs):
        self.lins, self.activation, optimizer,  self.batch_size, self.lr, self.indata, self.truth, self.cost =args
        
        keys = ('convs', 'csizes','lr_decay', 'max_epochs', 'saving', 'run_num', 'new_tar',
                'lr_min','dropout',  'plot_ev', 'print_ev', 'resnet', 'train_fac')
        for kwarg in kwargs.keys():
            if kwarg not in keys:
                raise Exception(kwarg + ' is not a valid key. Valid keys: ', keys )
        
        self.convs = kwargs.get('convs', [])
        self.csizes = kwargs.get('csizes', [])
        self.resnet = kwargs.get('resnet', False)
        if self.csizes and self.convs :
            if len(self.csizes) != len(self.convs):
                raise Exception('Number of convolutions does not match the number of convolution sizes')
        elif self.csizes or self.convs:
            raise Exception('Provide the number of convolution channels and the size of the convolution matrix')

        
        self.plotev = kwargs.get('plot_ev', 3)
        self.printev = kwargs.get('print_ev', 1)
        self.train_fac = kwargs.get('train_fac', 4/5)
        
        
        self.data = self.data_prep()
        self.datadims = len(self.indata.shape) - 1
        if self.datadims != 2 and self.convs:
            raise Exception('convolutions are only supported for 2d data')

                
        self.dropout = kwargs.get('dropout', 0)
    
        self.model = Net(self.indata.shape[1:], self.convs, self.lins,  self.csizes, self.dropout, 
                             self.resnet, self.activation).double()
        
        self.lr_decay = kwargs.get('lr_decay', 1)
        self.lr_min = kwargs.get('lr_min', 1e-10)
        self.optimizer = optimizer(self.model.parameters(), lr = self.lr)
        self.max_epochs = kwargs.get('max_epochs', 10)
            
        self.trainerr, self.testerr, self.err, self.epoch, self.loc, self.save_file = [], [], 0, 0, 0, None #just inits
        
        self.saving = kwargs.get('saving', False)
        
        self.new_tar = kwargs.get('new_tar', False)
        self.run_num = kwargs.get('run_num', None)
        self.check()

        
        if self.run_num:
            vals = pickle.load(open('values.p','rb')).loc[self.run_num]
            self.epoch = vals['epochs']
            self.err = vals['err']
            self.lr = vals['learning rate']
            print('Weights from run ', str(self.run_num), ' loaded.')
        else: self.epoch = 0        
            
        
    def params(self, re = True):
            
        if len(self.testerr)>1:
            derr = round(self.testerr[-2] - self.testerr[-1], 3)
        else:
            derr = 0
        
            
        df = pd.DataFrame(columns = ('convolution layer sizes', 'convolution matrix sizes',  
                                      'ResNet','linear layer sizes','activation', 'training sample size', 'learning rate',  
                             'Dropout',  'batch size',   
                            'epochs',  'derr', 'err', ))
        vals = ( self.convs, self.csizes,  self.resnet , self.lins, str(self.activation),
                int(self.train_fac*self.indata.shape[0]), self.lr, self.dropout,  self.batch_size,  
                    self.epoch + len(self.testerr),  derr, round(float(self.err),2),)
        df.loc[self.loc] = vals
        if re == True:
            return df
        else: display(df)
        
                
    def run(self, **kwargs):
        if len(self.trainerr)!=len(self.testerr):
            self.trainerr = self.trainerr[:-1]
        keys = ('lr', 'lr_decay', 'max_epochs', 'saving', 'batch_size', 'lr_min', 'training',)
        for kwarg in kwargs.keys():
            if kwarg not in keys:
                raise Exception(kwarg + ' is not a valid key. Valid keys: ', keys )
                
        train, traintruth, test, testtruth = self.data
        self.testtruth = testtruth
        self.max_epochs = kwargs.get('max_epochs', self.max_epochs)
        self.lr = kwargs.get('lr', self.lr)
        self.saving = kwargs.get('saving', self.saving)
        self.lr_decay = kwargs.get('lr_decay', self.lr_decay)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.lr_min = kwargs.get('lr_min', self.lr_min)
        training = kwargs.get('training', True)
        self.printev = kwargs.get('print_ev', self.printev)
        self.plotev = kwargs.get('plot_ev', self.plotev)
        e0 = self.epoch
        
        while self.epoch < (e0 + self.max_epochs) :
            shuffle = torch.randperm(len(traintruth)) #shuffle training set
            self.lr = max(self.lr * self.lr_decay, self.lr_min)  #lr decay
            for i in range(round(len(traintruth)/self.batch_size)):
                self.optimizer.param_groups[0]['lr'] = self.lr
                where = shuffle[i * self.batch_size:(i + 1) * self.batch_size] #take batch of training set
                self.output = self.model(train[where])
                self.truetrain = traintruth[where]
                loss = torch.mean(self.cost(torch.squeeze(self.output), torch.tensor(traintruth[where]).double()))
                if training:
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph = True)
                    self.optimizer.step()
                         
            self.trainerr.append(loss.detach().numpy())
            self.testout = self.model(test)
            self.err = self.cost(torch.squeeze(self.testout), torch.tensor(testtruth)
                                         ).mean().detach().numpy()
            self.testerr.append(self.err)
            if not (self.epoch-e0) % self.plotev:
                self.plot()
                if not training:
                    return []
            #if it's not training, stop
            l10 = self.testerr[-10:]
            if len(self.testerr)>10 and np.all(np.abs(l10 - np.mean(l10))<.005*np.mean(l10)) and l10[-1]>l10[0]:
                print('Not improving')
                break
            elif len(self.testerr)>10 and np.sum(l10[1:]>l10[:-1])>8:
                print('Overfitting')
                break
            if not (self.epoch-e0) % self.printev:
                print("Epoch number {}\n Current loss {}\n".format(self.epoch, self.err))
            if self.saving:
                self.save()
            self.epoch += 1
            

    def save(self): 
        torch.save(self.model.state_dict(), self.save_file)
        vals = pickle.load(open('values.p', 'rb'))
        df = self.params()
        if self.loc in vals.index.tolist():
            vals.loc[self.loc] = df.loc[self.loc]
        else:
            vals = vals.append( df)
        pickle.dump(vals, open('values.p', 'wb'))
        
    def check(self):
        
        df = self.params()
        vals = pickle.load(open('values.p', 'rb'))
        
        ## print warning and run params if current parameters match other runs
        same = np.array([])
        for i in list(vals.index):
            if vals.loc[i][:5].equals(df.loc[0][:5]):
                same = np.append(same, i)
                if not i==self.run_num:
                    display(pd.DataFrame(vals.loc[i]).T)
        if self.run_num and len(same):
            same = list(np.delete(same, np.where(same == self.run_num)))
        if len(same):
            warnings.warn('Run(s) '+ str(same) + ' used the same hyper parameters')
            
        # if saving to a new file, load weights. If continuing run, check that vals match
        if self.new_tar and type(self.run_num)==int:
            self.model.load_state_dict(torch.load('tars/' + str(self.run_num)+'.tar'))        
        elif not self.new_tar and self.run_num:
            if vals.loc[self.run_num][:5].equals(df.loc[self.loc][:5]):
                self.loc = self.run_num
                self.model.load_state_dict(torch.load('tars/' + str(self.loc)+'.tar'))   
            else: raise Exception('Parameters don\'t match')

                     
        if not self.loc: 
            self.loc = vals.index[-1] + 1
    
        if not self.saving:
            self.save_file = None
            print('Not saving')
        else: 
            self.save_file = 'tars/' + str(self.loc)+'.tar'
  
            
    def plot(self, med = True,):
        testtruth = self.data[-1]
        output = self.testout
        if len(self.testerr) > 1:
            fig, ax = plt.subplots(1,3, figsize = (15,4))
            xs = np.arange(len(self.trainerr))
            ax[2].plot(xs, self.trainerr, label = 'train error')
            ax[2].plot(xs, self.testerr, label = 'test error')
            ax[2].set_xlabel('epoch')
            ax[2].set_ylabel('mean abs err')
            ax[2].legend()
            ax[2].set_yscale('log')
        else: fig, ax = plt.subplots(1, 2, figsize = (15, 4))
            
        x, y = self.data[3], np.squeeze(np.squeeze(self.testout.detach().numpy()))

        ax[0].hist2d(x, y, bins = 30)

        ax[0].set_xlabel('test truth')
        ax[0].set_ylabel('test predication') 

        yt, xt = torch.squeeze(torch.squeeze(self.output)).detach().numpy(), self.truetrain
        ax[1].hist2d(xt, yt, bins = 30)
        ax[1].set_xlabel('train truth')
        ax[1].set_ylabel('train predication') 
        plt.show()
    
    def data_prep(self):
        data = torch.tensor(self.indata).double()
        
        if len(data.shape) == 1:
            data = torch.unsqueeze(data, 0)
        if len(data.shape) >= 2:
            data = torch.unsqueeze(data, 1)

        split = int(self.train_fac * self.truth.shape[0])
        train, traintruth = data[:split], self.truth[:split]
        test, testtruth = data[split:], self.truth[split:]
        return train, traintruth, test, testtruth