import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time

# expected daily return
def alpha_t(t,initial_strength,half_life):
    return initial_strength*1e-4*pow(2.0,-t/half_life)

# cost
def c_delta(delta, P, V, theta, sigma, gamma, eta, beta):
    X = delta / P
    return P * X * (0.5 * gamma * sigma * X / V * pow(theta / V,0.25) + np.sign(X) * eta * sigma * pow(np.abs(X / V),beta))

# golden section search
gr = (1.0 + math.sqrt(5.0)) / 2.0
def golden_section_search(f, a, b, tol):
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2.0

# find optimal trading path
def optimal_path(T,tol,P, V, theta, sigma, gamma, eta, beta,kappa,initial_strength,half_life):
    X=np.zeros(int(T)+1)
    t=1
    count=0
    while True:
        if count == T:
            break

        def func(x):
            delta1=x-X[t-1]
            if t==T:
                delta2=0
            else:
                delta2=X[t+1]-x
            return 0.5*kappa*sigma*sigma*x*x-x*alpha_t(t,initial_strength,half_life)+c_delta(delta1,P, V, theta, sigma, gamma, eta, beta)+c_delta(delta2,P, V, theta, sigma, gamma, eta, beta)
        if t==T:
            lb=X[t-1]-theta
            ub=X[t-1]+theta
        else:
            lb=max(X[t-1]-theta,X[t+1]-theta)
            ub=min(X[t-1]+theta,X[t+1]+theta)

        # golden section search
        x = golden_section_search(func,lb, ub, tol)

        if abs(x - X[t]) <= tol:
            count += 1
        else:
            count = 0
            X[t]=x

        if t==T:
            t=1
        else:
            t=t+1
    return X

def test1():
    # set parameters
    T = 30.0
    P = 40.0
    V = 2e6
    theta = 2e8
    sigma = 0.02
    initial_strength=50.0
    half_life=5.0

    # Almgren model
    gamma = 0.314
    eta = 0.142
    beta = 0.6

    # risk aversion
    kappa = 1e-7

    # optimize
    tol=1.0
    t1=time.clock()
    X=optimal_path(T,tol,P, V, theta, sigma, gamma, eta, beta,kappa,initial_strength,half_life)
    t2=time.clock()
    print "computation time: ",t2-t1
    res=pd.DataFrame()
    res["Dollar Position"]=X
    print res.to_string()

    plt.figure()
    plt.plot(X)
    plt.xlabel('Day')
    plt.ylabel('Dollar Position')
    plt.show()

def profit(X,T,P, V, theta, sigma, gamma, eta, beta,initial_strength,half_life):
    def c(delta):
        return c_delta(delta, P, V, theta, sigma, gamma, eta, beta)
    ts=range(1,int(T)+1)
    alphas=map(lambda x:alpha_t(x,initial_strength,half_life),ts)
    rtn=np.sum(X[1:]*alphas)
    deltas=np.diff(X)
    cost=np.sum(map(c,deltas))
    return rtn-cost

def sharpe_ratio(X,T,P, V, theta, sigma, gamma, eta, beta,initial_strength,half_life):
    vol=math.sqrt(np.sum(map(lambda x:x**2,X)))*sigma
    return profit(X,T,P, V, theta, sigma, gamma, eta, beta,initial_strength,half_life)/vol*math.sqrt(252.0)

def test2():
    T = 30.0
    P = 40.0
    V = 2e6
    theta = 2e8
    # Almgren model
    gamma = 0.314
    eta = 0.142
    beta = 0.6
    tol=1.0
    # parameters
    kappas = np.array([1e-8, .5 * 1e-7, 1e-7, .2 * 1e-6, .5 * 1e-6, 1e-6, .2 * 1e-5, .5 * 1e-5, 1e-5, 1e-4])
    half_lifes = np.arange(1, 10.5, .5)
    initial_strengths = np.arange(5, 105, 5)
    sigmas = np.arange(.01, .11, .01)

    # change kappa
    print "kappa"
    sigma = 0.02
    initial_strength = 50.0
    half_life = 5.0
    profits1=[]
    sharpe_ratios1=[]
    for kappa in kappas:
        X=optimal_path(T,tol,P, V, theta, sigma, gamma, eta, beta,kappa,initial_strength,half_life)
        profits1.append(profit(X,T,P, V, theta, sigma, gamma, eta, beta,initial_strength,half_life))
        sharpe_ratios1.append(sharpe_ratio(X,T,P, V, theta, sigma, gamma, eta, beta,initial_strength,half_life))
    print "finish"

    # change half life
    print "half life"
    sigma = 0.02
    initial_strength = 50.0
    kappa=1e-7
    profits2 = []
    sharpe_ratios2 = []
    for half_life in half_lifes:
        X = optimal_path(T, tol, P, V, theta, sigma, gamma, eta, beta, kappa, initial_strength, half_life)
        profits2.append(profit(X, T, P, V, theta, sigma, gamma, eta, beta, initial_strength, half_life))
        sharpe_ratios2.append(sharpe_ratio(X, T, P, V, theta, sigma, gamma, eta, beta, initial_strength, half_life))
    print "finish"

    # change initial strength
    print "initial strength"
    sigma = 0.02
    half_life = 5.0
    kappa = 1e-7
    profits3 = []
    sharpe_ratios3 = []
    for initial_strength in initial_strengths:
        X = optimal_path(T, tol, P, V, theta, sigma, gamma, eta, beta, kappa, initial_strength, half_life)
        profits3.append(profit(X, T, P, V, theta, sigma, gamma, eta, beta, initial_strength, half_life))
        sharpe_ratios3.append(sharpe_ratio(X, T, P, V, theta, sigma, gamma, eta, beta, initial_strength, half_life))
    print "finish"

    # change sigma
    print "sigma"
    initial_strength=50.0
    half_life = 5.0
    kappa = 1e-7
    profits4 = []
    sharpe_ratios4 = []
    for sigma in sigmas:
        X = optimal_path(T, tol, P, V, theta, sigma, gamma, eta, beta, kappa, initial_strength, half_life)
        profits4.append(profit(X, T, P, V, theta, sigma, gamma, eta, beta, initial_strength, half_life))
        sharpe_ratios4.append(sharpe_ratio(X, T, P, V, theta, sigma, gamma, eta, beta, initial_strength, half_life))
    print "finish"

    plt.figure()
    plt.plot(kappas, profits1)
    plt.xlabel('kappa')
    plt.ylabel('profit')
    plt.figure()
    plt.plot(kappas, sharpe_ratios1)
    plt.xlabel('kappa')
    plt.ylabel('sharpe_ratio')

    plt.figure()
    plt.plot(half_lifes, profits2)
    plt.xlabel('half_life')
    plt.ylabel('profit')
    plt.figure()
    plt.plot(half_lifes, sharpe_ratios2)
    plt.xlabel('half_life')
    plt.ylabel('sharpe_ratio')

    plt.figure()
    plt.plot(initial_strengths, profits3)
    plt.xlabel('initial_strength')
    plt.ylabel('profit')
    plt.figure()
    plt.plot(initial_strengths, sharpe_ratios3)
    plt.xlabel('initial_strength')
    plt.ylabel('sharpe_ratio')

    plt.figure()
    plt.plot(sigmas, profits4)
    plt.xlabel('sigma')
    plt.ylabel('profit')
    plt.figure()
    plt.plot(sigmas, sharpe_ratios4)
    plt.xlabel('sigma')
    plt.ylabel('sharpe_ratio')

    plt.show()

if __name__ == "__main__":
    ### uncomment each function to test
    #test1()
    test2()