import numpy as np

data = np.loadtxt('data1000.txt',delimiter=',').astype(int)
dim = 10
V = data[:,1:dim+1].copy()
per = 0.07
N = len(V)
K = int(N*per)

group = np.unique(V[:,0])
per_low = 0.05
per_up = 0.15
P = {}

s_meet_v = lambda S, V: len([item for item in S if item in V])

for item in group:
    P[item] = {}
    index = list(np.where(V[:, 0] == item)[0])
    P[item]['index'] = index
    P[item]['low'] = np.ceil(len(index)*per_low).astype(int)
    P[item]['up'] = np.ceil(len(index)*per_up).astype(int)

all_index = np.arange(N)
Error22 = []
object22 = []
for i in range(100):
    random_SS = np.random.choice(all_index, K, replace=False)
    Error2 = 0
    for item in P:
        meet_num = s_meet_v(random_SS, P[item]['index'])
        if meet_num < P[item]['low']:
            Error2 += P[item]['low'] - meet_num
        elif meet_num > P[item]['up']:
            Error2 += meet_num - P[item]['up']
        else:
            continue
    Error22.append(Error2)

print(np.mean(Error22))

