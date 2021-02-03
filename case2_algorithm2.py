import numpy as np
import multiprocessing as mp
from sklearn.metrics import pairwise_distances

dim = 10 # Data dimension
per = 0.15 #  The proportion of cardinal number
k = 0.3 #  Curvature

data = np.loadtxt('data1000.txt',delimiter=',').astype(int)
V = data[:,1:dim+1].copy()
K = int(len(V)*per)
N = len(V)

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


def M(item):
    return H(k)*(G(item)+alpha*L(item))

s_meet_v = lambda S, V: len([item for item in S if item in V])

H = lambda k: (1 - k) - np.sqrt(np.power((1 - k), 3) / (2 - k))
S = {}
epsilon = 0.1
alpha = 2 - k - np.sqrt((2 - k) * (1 - k))

if __name__ == '__main__':
    pool = mp.Pool()
    res = pool.map(M,V)
    pool.close()
    pool.join()
    m = max(res)
    low = m / ((1 + epsilon) * K)
    up = m / ((1 - k) * H(k))
    start = np.ceil(np.log(low) / np.log(1 + epsilon))
    end = np.ceil(np.log(up) / np.log(1 + epsilon))
    O = []
    for i in range(int(start), int(end)):
        O.append(np.power((1 + epsilon), i))
    for item in O:
        if item not in list(S.keys()):
            S[item] = []
    for i in range(len(V)):
        print(i)
        for item in S:
            if len(S[item]) >= K:
                continue
            temp = S[item].copy()
            temp.append(i)
            if G(V[temp]) - G(V[S[item]]) + alpha*L(V[i]) > item:
                S[item].append(i)
    key_list = list(S.keys())
    index = int(np.argmax([G(V[S[item]])+L(V[S[item]]) for item in S ]))
    SS = S[key_list[index]]
    object1 = G(V[SS]) + L(V[SS])
    Error1 = 0
    for item in P:
        meet_num = s_meet_v(SS,P[item]['index'])
        if meet_num < P[item]['low']:
            Error1 += P[item]['low'] - meet_num
        elif meet_num > P[item]['up']:
            Error1 += meet_num - P[item]['up']
        else:
            continue

    print('Error_algorithm2:',Error1)
    print('object_algorithm2:', object1)











