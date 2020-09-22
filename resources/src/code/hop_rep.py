import json
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from scipy.stats import bernoulli
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow import keras

""" 
    DESCRIPTION: 
        Basic Hopfield network model 
    USAGE: 
        python -u hop_rep.py --input_seqs=intputfile.json --test_seq=testseq.json --random --N=100 --M=10 
        if --random, input is ignored and M seq of length N are randomly generated and stored.
        inputfile must be a multijson file with 1 seq for each line. 
        testseq is a json with a sequence to be tested. 

    AUTHOR: 
        David Preti
    DATE: 
        Rome, Sep 2020 
"""
class network:
    def __init__(self, x, temp):
        self.x = x
        self.M = x.shape[0]
        self.N = x.shape[1]
        self.T = temp
        self.net = np.zeros(shape=(self.N,self.N))
        for mu in range(self.M):
            for i in range(self.N):
                for j in range(self.N):
                    self.net[i,j] += (x[mu,i]*x[mu,j])
                    if i==j:
                        self.net[i,j] = 0.0
        self.net  = (self.net)/self.N

    def update(self,S,iterations=100):
        def energy(s):
            E = 0 
            for i in range(self.N):
                for j in range(self.N):
                    E += self.net[i,j]*s[i]*s[j]
            return -0.5*E

        S = S[0,:]
        d = {}
        h = np.zeros(shape=(self.N))
        E = []
       
        E.append(energy(S))
        for mu in range(self.M):
            d[mu] = []
        for _ in range(iterations):
            n = random.randint(0,self.N-1)
            h[n] = 0
            for j in range(self.N):
                h[n] += self.net[n,j]*S[j]
                if self.T==0:
                    S = np.where(h<=0, -1, 1)
                else :
                    S = np.where([random.random() for _ in range(self.N)] <= 0.5*(1.0+np.tanh(h/T)), 1 , -1)
            E.append(energy(S))
            for mu in range(self.M):
                d[mu].append(hamming_distance(self.x[mu,:],S))
        return S,d,E

def random_sequence(N,M):
    seq = []
    for _ in range(M):
        seq.append([x if x==1 else -1 for x in bernoulli.rvs(p=0.5, size=N)])
    return np.array(seq)

def hamming_distance(x,y):
    return np.abs(np.sum(x-y))/len(x)

def best_pattern(d):
    last_d = []
    for mu in range(len(d.keys())):
        last_d.append(d[mu][-1])
    return np.argwhere(last_d == np.amin(last_d))

def plot_energy(e,temp):
    plt.plot(e)
    plt.title('Hopfiled T={}'.format(temp))
    plt.xlabel('t')
    plt.ylabel('E(t)')
    plt.show()

def plot_m(m,temp):
    plt.scatter(temp,m)
    plt.xlabel('T')
    plt.ylabel('m(T)')
    plt.show()

def plot_hamming(d,temp):
    for mu in range(len(d.keys())):
        plt.plot(d[mu],label=str(mu))
    plt.title('Hopfield T={}'.format(temp))
    plt.xlabel('t')
    plt.ylabel('hamming distance')
    plt.legend()
    plt.show()

#### MAIN
print("="*50)
print("="*50)
start_time = time.time()                   
now = datetime.now()
print(" --- STARTED : {} ---".format(now))
print("="*50)

parser = argparse.ArgumentParser()
parser.add_argument("--N", help = "pattern length (if --random)")
parser.add_argument("--M", help = "number of different patterns (if --random)")
parser.add_argument("--iter", help = "number of iterations")
parser.add_argument("--plot",help = "plot or not", action="store_true")
parser.add_argument("--replicas",help='number of replicas')

args = parser.parse_args()
ITERATIONS    = int(args.iter)
N             = int(args.N)
M             = int(args.M) 
isplot        = args.plot
replicas      = int(args.replicas)
distances     = {}
np.random.seed(seed=1)

x = random_sequence(N=N,M=M)
m_T = []
for T in tqdm(np.arange(0,2,0.1)):
    S_average = np.zeros(shape=(N))
    m = []
    for n in range(replicas):
        S = random_sequence(N=N,M=1)
        if S.shape[1]!=x.shape[1]:
            print('error')
            exit(1)
            
        net = network(x,temp=T)
        S_updated,distances,energy = net.update(S=S,iterations=ITERATIONS)
        S_average += S_updated
    for mu in range(M):
        tmp = 0 
        for i in range(N):
            tmp += S_average[i]*x[mu,i]
        m.append(tmp) 
    m_T.append(np.min(m)/replicas/N)


print("="*50)
tot_time = time.time() - start_time
now = datetime.now()
print (' --- DONE    : total time: {} sec ---'.format(tot_time))
print("="*50)

if isplot:
    plot_m(m_T,np.arange(0,2,0.1))
    