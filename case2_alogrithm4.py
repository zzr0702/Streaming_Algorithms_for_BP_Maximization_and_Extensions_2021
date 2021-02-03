import numpy as np
import multiprocessing as mp
from sklearn.metrics import pairwise_distances

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
    P[item]['low'] = np.floor(len(index)*per_low).astype(int)
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
    return N*N

H = lambda k: (2 - k) / (1 - k)

if __name__ == '__main__':
    S = []
    U = {}
    alpha = 1
    all_index = list(np.arange(len(V)))
    ii = 0
    for u in all_index:
        ii += 1
        print(ii)
        S_temp = S.copy()
        S_temp.append(u)
        if if_extendable(S_temp):
            U[u] = G(V[S_temp]) - G(V[S])
            S.append(u)
        else:
            U_temp = []
            for item in S:
                S_temp.remove(item)
                if if_extendable(S_temp):
                    U_temp.append(item)
                S_temp.append(item)
            index_value = [U[i] for i in U_temp]
            u_exchange = U_temp[np.argmin(index_value)]
            u_exchange_value = np.min(index_value)
            if G(V[S_temp]) - G(V[S]) + alpha * L(V[u]) >= H(k) * (u_exchange_value + alpha * L(V[u_exchange])):
                U[u] = G(V[S_temp]) - G(V[S])
                S.remove(u_exchange)
                S.append(u)
    if len(S) == K:
        object =  G(V[S]) + L(V[S])
    else:
        for item in P:
            can_add = P[item]['up'] - s_meet_v(P[item]['index'],S)
            if can_add != 0:
                can_add_u = list(set(P[item]['index'])-set(S))
                S = S + can_add_u[:int(min(can_add,K-len(S)))]
                if len(S) == K:
                    break
    object = G(V[S]) + L(V[S])
    print(object)

