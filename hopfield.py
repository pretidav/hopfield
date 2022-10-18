import json
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from scipy.stats import bernoulli
import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

""" 
    DESCRIPTION: 
        Basic Hopfield network model 
    USAGE: 
        python -u hop_rep.py --input_seqs=intputfile.json --test_seq=testseq.json --random --N=100 --M=10  --plot --threads=4
        if --random, input is ignored and M seq of length N are randomly generated and stored.
        inputfile must be a multijson file with 1 seq for each line. 
        testseq is a json with a sequence to be tested. 

    AUTHOR: 
        David Preti
    DATE: 
        Rome, Oct 2022 
"""
class HopfieldNetwork:
    def __init__(self, x, temp):
        self.x = x
        self.M = x.shape[0]
        self.N = x.shape[1]
        self.eps = np.finfo(float).eps
        self.T = temp
        self.net = self.initialize_net(N=self.N,M=self.M)
    
    def initialize_net(self,N,M):
        net = np.zeros(shape=(N,N))
        for mu in range(M):
            for i in range(N):
                for j in range(N):
                    net[i,j] += (self.x[mu,i]*self.x[mu,j])/(float(N)*float(M))
                    if i==j:
                        net[i,j] = 0.0
        return net

    def hamming_distance(self,x,y):
        return np.abs(np.sum(x-y))/len(x)

    def get_energy(self,s):
        E = 0 
        for i in range(self.N):
            for j in range(self.N):
                E += self.net[i,j]*s[i]*s[j]
        return -1.* E / self.N  # energy is normalized to 1 

    def update(self,S,iterations=100):
        s = S[0,:]
        E = []
        d = {mu: [] for mu in range(self.M)}
        E.append(self.get_energy(s))
        for i in range(iterations):
            n = list(range(0,self.N))
            random.shuffle(n)
            h = np.zeros(shape=(self.N))
            for j in n:
                h[j] = np.sum(self.net[j,:]*s[:])
                s = np.where( np.random.uniform(low=0,high=1,size=self.N) <= (1.0+np.tanh(h/(self.T+self.eps))), 1 , -1)
            E.append(self.get_energy(s))
            [d[mu].append(self.hamming_distance(self.x[mu,:],s)) for mu in range(self.M)] # distance from stored patterns * iterations
        return s,d,E

def random_sequence(N,M):
    seq = []
    for _ in range(M):
        seq.append([x if x==1 else -1 for x in bernoulli.rvs(p=0.5, size=N)])
    return np.array(seq)

def best_pattern(d):
    last_d = []
    for mu in range(len(d.keys())):
        last_d.append(d[mu][-1])
    ar = np.argwhere(last_d == np.amin(last_d))[0]
    if len(ar)>1: 
        ar = ar[-1] 
    return ar

def plot_energy(e,temp,N,M,pathfig):
    e = np.array(e)
    e_mean = np.mean(e,axis=1)
    e_err  = np.std(e,axis=1)/np.sqrt(e.shape[1])  
    x = list(range(0,e.shape[2]))

    for i in range(e.shape[0]):
        plt.plot(x,e_mean[i,:],label="T={}".format(temp[i]))
        plt.fill_between(x, e_mean[i,:]-e_err[i,:], e_mean[i,:]+e_err[i,:], alpha=0.4)
        
    plt.title('Hopfield (N,M)=({},{})'.format(N,M))
    plt.xlabel('t')
    plt.ylabel('<E(t)>')
    plt.legend(bbox_to_anchor=(1., 1.), loc="upper left")
    plt.savefig(pathfig+"energy_t_"+str(N)+"-"+str(M)+".png", bbox_inches="tight")

    plt.clf()
    fig = plt.figure()
    fig.suptitle('Hopfield (N,M)=({},{})'.format(N,M))
    gs = fig.add_gridspec(2, hspace=0)
    (ax1, ax2) = gs.subplots(sharex=True)
    for i in range(e.shape[0]):
        ax1.errorbar(x=temp[i],y=e_mean[i,-1],yerr=e_err[i,-1],fmt='o', markersize=6, capsize=6,color='k')        
    ax1.set_ylabel('<E(T)>')
    for i in range(e.shape[0]):
        ax2.scatter(x=temp[i],y=e_err[i,-1],marker='o',color='k')        
    ax2.set_xlabel('T')
    ax2.set_ylabel('<dE(T)>')
    plt.savefig(pathfig+"energy_T_"+str(N)+"-"+str(M)+".png", bbox_inches="tight")
    plt.clf() 

def plot_m(m,b,temp,pathfig,N,M):
    for i in range(len(temp)): 
        plt.scatter(x=temp[i],y=m[i][b[i]],marker='o',color='b')
    plt.title('Hopfield (N,M)=({},{})'.format(N,M))
    plt.xlabel('T')
    plt.ylabel('<m(T)>')
    plt.savefig(pathfig+"m_T_"+str(N)+"-"+str(M)+".png", bbox_inches="tight")

def plot_hamming(d,temp):
    for mu in range(len(d.keys())):
        plt.plot(d[mu],label=str(mu))
    plt.title('Hopfield T={}'.format(temp))
    plt.xlabel('t')
    plt.ylabel('hamming distance')
    plt.legend()


if __name__=='__main__':
    print("="*50)
    print("="*50)
    start_time = time.time()                   
    now = datetime.now()
    print(" --- STARTED : {} ---".format(now))
    print("="*50)

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help = "pattern length (if --random)",default=5)
    parser.add_argument("--M", help = "number of different patterns (if --random)",default=2)
    parser.add_argument("--iter", help = "number of iterations",default=10)
    parser.add_argument("--plot",help = "plot or not", action="store_true")
    parser.add_argument("--replicas",help='number of replicas',default=1)
    parser.add_argument("--threads", help='number of parallel threads',default=1)
    parser.add_argument("--seed", help='Seed',default=1988)
    parser.add_argument("--Tmin",help='minimum T',default=0)
    parser.add_argument("--Tmax",help='maximum T',default=0.1)
    parser.add_argument("--Tby",help='steps T',default=0.1)
    parser.add_argument("--temp",help='T',default=0)

    args = parser.parse_args()
    ITERATIONS    = int(args.iter)
    N             = int(args.N)
    M             = int(args.M) 
    isplot        = args.plot
    replicas      = int(args.replicas)
    threads       = int(args.threads)
    temp          = float(args.temp)
    Tmin          = float(args.Tmin)
    Tmax          = float(args.Tmax)
    Tby           = float(args.Tby)
    np.random.seed(seed=int(args.seed))

    distances     = {}

    x = random_sequence(N=N,M=M)
    S = random_sequence(N=N,M=1)
    m_T = []
    e_T = []
    best_class_T = []

    def func(T,N=N,M=M,replicas=replicas,x=x):        
        S_average = np.zeros(shape=(N))
        m = []
        b = []
        e = []
        for r in range(replicas):
            S = random_sequence(N=N,M=1)
            if S.shape[1]!=x.shape[1]:
                print('error {}!={}'.format(S.shape[1],x.shape[1]))
                exit(1)
            net = HopfieldNetwork(x,temp=T)
            S_updated,distances,energy = net.update(S=S,iterations=ITERATIONS)
            S_average += S_updated/float(replicas)
            e.append(energy)
            b.append(best_pattern(d=distances))      
        for mu in range(M):
            m.append(np.sum(S_average[:]*x[mu,:]))
        mean_b = int(np.mean([i for i in b]))
        return m,mean_b,e

    with Pool(threads) as p:
        A=list(tqdm(p.imap(func, list(np.arange(Tmin,Tmax,Tby))),total=len(list(np.arange(Tmin,Tmax,Tby)))))
    for (m,d,e) in A:
        m_T.append(m)
        best_class_T.append(d)
        e_T.append(e)

    if isplot:
        plot_energy(e=e_T,temp=np.arange(Tmin,Tmax,Tby), N=N, M=M,pathfig='./figs/')    
        plot_m(m=m_T,b=best_class_T,temp=np.arange(Tmin,Tmax,Tby),N=N,M=M, pathfig='./figs/')

    print("="*50)
    tot_time = time.time() - start_time
    now = datetime.now()
    print (' --- DONE    : total time: {} sec ---'.format(tot_time))
    print("="*50)

