import numpy as np
from sklearn.metrics import pairwise_distances
import time

np.set_printoptions(suppress=True)



dim = 10 # Data dimension
per = 0.15 #  The proportion of cardinal number
k = 0.3 #  Curvature

data = np.loadtxt('data1000.txt',delimiter=',').astype(int)
V = data[:,1:dim+1].copy()
K = int(len(V)*per)
N = len(V)


s_meet_v = lambda S,V :len([item for item in S if item in V])

group = np.unique(V[:,0])
per_low = 0.05
per_up = 0.15
P = {}

for item in group:
    P[item] = {}
    index = list(np.where(V[:, 0] == item)[0])
    P[item]['index'] = index
    P[item]['low'] = np.ceil(len(index)*per_low).astype(int)
    P[item]['up'] = np.ceil(len(index)*per_up).astype(int)

def if_extendable(solution):
    S = solution.copy()
    total = 0
    for item in P.keys():
        if s_meet_v(S,P[item]['index']) > P[item]['up']:
            return False
        else:
            total += max([s_meet_v(S,P[item]['index']),P[item]['low']])
    if total > K:
        return False
    return True

distances_max = np.max(pairwise_distances(V.reshape(-1,dim),V.reshape(-1,dim),metric= 'euclidean'))
def G(S):
    if len(S) == 0:
        return 0
    total_distance = distances_max*len(V)

    N = len(S)
    M = len(V)
    distance = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            if i == j:
                distance[i][j] = np.inf
            distance[i][j] = np.sqrt(sum(np.power(S[i] - V[j], 2)))
    S_distance = np.sum(np.amin(distance,axis=0))
    f = total_distance - S_distance
    return f


def L(S):
    if len(S) == 0:
        return 0
    if np.shape(S)[0] == dim:
        S = S[np.newaxis, :]
    N = np.shape(S)[0]
    return 100*N*N

if __name__ == '__main__':
    S = []
    alpha = 2 / (2 - k)
    all_index = list(np.arange(len(V)))
    while len(S) < K:
        print(len(S))
        U = [[],[]]
        for i in all_index:
            S_temp = S.copy()
            S_temp.append(i)
            if if_extendable(S_temp):
                U[0].append(i)
                value = G(V[S_temp]) - G(V[S]) + alpha*L(V[i])
                U[1].append(value)
        add = U[0][np.argmax(U[1])]
        S.append(add)
        all_index.remove(add)
    object = G(V[S]) + L(V[S])
    print(object)




