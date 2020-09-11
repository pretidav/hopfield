import json
import argparse
import time
from datetime import datetime
#from tqdm import tqdm
from scipy.stats import bernoulli
import numpy as np
import random
import matplotlib.pyplot as plt

""" 
    DESCRIPTION: 
        Basic Hopfield network model 
    USAGE: 
        python -u hop.py --input_seqs=intputfile.json --test_seq=testseq.json --random --N=100 --M=10 
        if --random, input is ignored and M seq of length N are randomly generated and stored.
        inputfile must be a multijson file with 1 seq for each line. 
        testseq is a json with a sequence to be tested. 

    AUTHOR: 
        David Preti
    DATE: 
        Rome, Sep 2020 
"""
class network:
    def __init__(self, x):
        self.x = x
        self.M = x.shape[0]
        self.N = x.shape[1]
        self.net = np.zeros(shape=(self.N,self.N))
        for mu in range(self.M):
            for i in range(self.N):
                for j in range(self.N):
                    self.net[i,j] += x[mu,i]*x[mu,j] 
        self.net  = (self.net)/self.M
        self.net -= np.eye(self.N)

    def update(self,S,iterations=10):
        S = S[0,:]
        h = np.zeros(shape=(self.N))
        d = {}
        for mu in range(self.M):
            d[mu] = []
        for t in range(iterations):
            S_old = S.copy()
            for i in range(self.N):
                for j in range(self.N):
                    h[i] += self.net[i,j]*S[j]
                h[i] = np.sign(h[i])
            for i in range(self.N):
                if h[i]>0:
                    S[i]=1
                else :
                    S[i]=-1
            for mu in range(self.M):
                d[mu].append(hamming_distance(self.x[mu,:],S))
            if (S==S_old).all():
                print('converged at t = {}'.format(t))
                break
        return S,d

def read_multijson(INPUT):
    seq = []
    total = 0
    with open(INPUT,'r',encoding='utf8') as file:
        for line in file:
            total += 1
            line = line.strip()
            ss = json.loads(line)
            seq.append(ss['seq'])
    return np.array(seq)

def random_sequence(N,M):
    seq = []
    for _ in range(M):
        seq.append([x if x==1 else -1 for x in bernoulli.rvs(p=0.5, size=N)])
    return np.array(seq)

def hamming_distance(x,y):
    return np.abs(np.sum(x-y))

def best_pattern(d):
    last_d = []
    for mu in range(len(d.keys())):
        last_d.append(d[mu][-1])
    return np.argwhere(last_d == np.amin(last_d))

def plot_hamming(d):
    for mu in range(len(d.keys())):
        plt.plot(d[mu],label=str(mu))
    plt.title('Hopfield T={}'.format(0))
    plt.xlabel('time')
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
parser.add_argument("--input_seq", help = "multijson with sequences to be stored")
parser.add_argument("--test_seq", help = "json with sequence to be tested")
parser.add_argument("--N", help = "pattern length (if --random)")
parser.add_argument("--M", help = "number of different patterns (if --random)")
parser.add_argument("--random", help = "ignore input and create random patterns", action="store_true")
args = parser.parse_args()
PATH_IN       = args.input_seq
TEST_IN       = args.test_seq
N             = int(args.N)
M             = int(args.M) 
israndom      = args.random 
distances     = {}
np.random.seed(seed=1)


if israndom:
    x = random_sequence(N=N,M=M)
    S = random_sequence(N=N,M=1)
    print('Storing {} random patterns of length {}'.format(M,N))
    print('Generating test sequence')
else :
    x = read_multijson(INPUT=PATH_IN)
    S = read_multijson(INPUT=TEST_IN)
    print('Storing {} patterns {} of length from {}'.format(len(x),len(x[0]),PATH_IN))
    print('Test seq from {}'.format(TEST_IN))

if S.shape[1]!=x.shape[1]:
    print('error')
    exit(1)
    
print(x)
print('INPUT SEQ : {}'.format(S))
net = network(x)
S_updated,distances = net.update(S=S)
print('OUTPUT SEQ: {}'.format(S_updated))
best = best_pattern(d=distances)
print(best[:,0])
for b in best[:,0]:
    print('closest patterns {} with distance {}'.format(b,distances[b][-1]))

plot_hamming(d=distances)
print("="*50)
tot_time = time.time() - start_time
now = datetime.now()
print (' --- DONE    : total time: {} sec ---'.format(tot_time))
print("="*50)
