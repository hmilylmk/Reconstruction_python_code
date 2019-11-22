import numpy as np
from scipy import sparse
import networkx as nx
from scipy import stats


def BA(N, m0, K, add_K):
    """建立无标度网；N为节点数，m0为初始节点数，K为初始规则网的平均度值，add_K为加边度值。"""
    m = np.zeros([N, N])
    for i in range(m0):
        for j in range(m0):
            if (j > i and j <= (i + K)):
                m[i, j] = 1
                m[j, i] = 1
            if ((i + K) > m0 and j <= (j + K -m0)):
                m[i, j] = 1
                m[j, i] = 1
    for step in range(m0, N):
        allnodes_k = np.sum(m, axis=0)
        total_k = np.sum(np.sum(m, axis=0))
        addi = 0
        while addi < add_K:
            rand_num = np.random.uniform()
            sum_k = 0
            select = 1
            for i in range(step):
                if (sum_k / total_k) <= rand_num and ((sum_k + allnodes_k[i]) / total_k) > rand_num:
                    select = i
                    break
                sum_k = sum_k + allnodes_k[i]
            if m[step, select] == 0:
                m[step, select] = 1
                m[step, select] = 1
                addi = addi + 1
    return m

def SIR(A, origin, beta):
    """
    参数:
        A: 网络的邻接矩阵
        origin: 传播源
        beta: 每一步的传播概率
    
    返回值:
        coverage: 传播结束时所有被感染的节点
        speed: 每一步的传播速度（每一个感染了多少节点）
        t: 每个节点被感染时的步长，从1开始连续的整数


    """
    N = A.shape[0]
    judge = 1
    S = np.arange(N)
    S = np.delete(S, origin)
    S.astype(int)
    I = []
    I.append(origin)
    R = []
    t = []
    h = 1
    speed = [0] * (N+1)
    while judge == 1:
        temp_I = []
        M = len(I)
        for j in range(M):
            node = int(I[j])
            asd = np.where(A[node,:] == 1.)[1]
            asd2 = np.intersect1d(asd, S)
            asd2.astype(int)
            num = np.random.rand(1, asd2.shape[0]) - beta
            asd3 = np.where(num <= 0)[1]
            asd3.astype(int)
            asd_final = asd2[asd3]
            temp_I = np.union1d(temp_I, asd_final)
            temp_I = temp_I.astype(np.int32)
            S = np.setdiff1d(S, asd_final)
            S.astype(int)
        if len(I) > 0:
            I = np.array(I)
            I.astype(int)
            R.extend(I)
            t.extend([h] * len(I))
        I = temp_I
        I.astype(int)
        if len(I) == 0:
            judge = 0
        speed[h] = len(R)
        h += 1
    coverage = R
    speed[h:] = [speed[h-1]] * (N - h + 1)
    return coverage, speed, t

def SI(A, origin, beta):
    N = A.shape[0]
    judge = 1
    S = np.arange(N)
    S = np.delete(S, origin)
    I = []
    I.append(origin)
    R = []
    t = []
    h = 1
    infected = []
    infected[:] = I[:]
    speed = [0] * (N + 1)
    while judge == 1:
        temp_I = []
        M = len(infected)
        for j in range(M):
            node = int(infected[j])
            asd = np.where(A[node,:] == 1.)[1]
            asd2 = np.intersect1d(asd, S)
            asd2.astype(int)
            num = np.random.rand(1, asd2.shape[0]) - beta
            asd3 = np.where(num <= 0)[1]
            asd3.astype(int)
            asd_final = asd2[asd3]
            temp_I = np.union1d(temp_I, asd_final)
            temp_I = temp_I.astype(np.int32)
            S = np.setdiff1d(S, asd_final)
            S.astype(int)
        if len(I) > 0:
            I = np.array(I)
            I.astype(int)
            R.extend(I)
            t.extend([h] * len(I))
        I = temp_I
        I.astype(int)
        infected = np.union1d(infected, I)
        if len(S) == 0:
            judge = 0
        speed[h] = len(R)
        h += 1
    coverage = R
    speed[h:] = [speed[h-1]] * (N - h + 1)
    return coverage, speed, t

def LT(A, origin, beta):
    N = A.shape[0]
    judge = 1
    S = np.arange(N)
    S = np.delete(S, origin)
    S_threshold = beta

    S_seed = []
    S_seed.append(origin)
    I = []
    I.append(origin)
    R = []
    t = []
    h = 1
    infected = []
    infected[:] = S_seed[:]
    speed = [0] * (N + 1)
    while judge == 1:
        temp_I = []
        M = len(infected)
        for j in range(M):
            node = int(infected[j])
            asd = np.where(A[node,:] == 1)[1]
            asd2 = np.intersect1d(asd, S)
            asd2.astype(int)
            number = np.array([])

            for k in range(len(asd2)):
                asd1 = np.where(A[:, asd2[k]] == 1)
                asd12 = np.intersect1d(asd1, infected)
                if len(asd12) * (1/len(asd1)) >= S_threshold:
                    number = np.union1d(number, k)
            number = number.astype(int)
            asd_final = asd2[number]
            temp_I = np.union1d(temp_I, asd_final)
            temp_I = temp_I.astype(np.int32)
            S = np.setdiff1d(S, asd_final)
            S.astype(int)
        if len(I) > 0:
            I = np.array(I)
            I.astype(int)
            R.extend(I)
            t.extend([h] * len(I))
        I = temp_I
        I.astype(int)
        infected = np.union1d(infected, I)
        if len(I) == 0:
            judge = 0
        speed[h] = len(R)
        h += 1
    coverage = R
    speed[h:] = [speed[h-1]] * (N - h + 1)
    return coverage, speed, t

def calculate_salton_similarity(record, N):
    """
    参数:
        record: 所有传播源完成传播后的矩阵，形状是[virus, N], virus是有多少个传播源参与传播， N是网络的节点总数，
                record[i, j] = 1表示网络中第j个节点被第i个传播源感染了， = 0表示没被该传播源感染。
        N: 网络的节点总数
    
    返回值:
        sim: 通过传播信息得到的网络节点对的相似度值矩阵。
             sim[i, j]表示节点i和节点j之间的相似度
    
    之后的计算相似度的函数都与之类似。

    """
    nn = record.shape[0]
    net = np.zeros([N, nn])
    for i in range(nn):
        id = record[i, np.where(record[i, :] > 0)]
        id = id.astype(int)
        net[id, i] = 1
    sim = np.dot(net, net.T)
    kout = np.sum(net.T, axis=0)
    oh = sparse.dia_matrix(([kout], [0]), shape=(N,N))
    matrix = oh.dot(np.ones((N,N)))
    denominator = (matrix * (matrix.T)) ** 0.5
    for i in range(N):
        for j in range(N):
            if denominator[i, j] != 0:
                sim[i, j] = sim[i, j] / denominator[i, j]
            else:
                sim[i, j] = 0
    return sim

def calculate_Sorensen_similarity(record, N):
    nn = record.shape[0]
    net = np.zeros([N, nn])
    for i in range(nn):
        id = record[i, np.where(record[i, :] > 0)]
        id = id.astype(int)
        net[id, i] = 1
    sim = np.dot(net, net.T)
    kout = np.sum(net.T, axis=0)
    oh = sparse.dia_matrix(([kout], [0]), shape=(N, N))
    matrix = oh.dot(np.ones((N,N)))
    denominator = matrix + (matrix.T)
    for i in range(N):
        for j in range(N):
            if denominator[i, j] != 0:
                sim[i, j] = sim[i, j] / denominator[i, j]
            else:
                sim[i, j] = 0
    return sim

def calculate_HPI_similarity(record, N):
    nn = record.shape[0]
    net = np.zeros([N, nn])
    for i in range(nn):
        id = record[i, np.where(record[i, :] > 0)]
        id = id.astype(int)
        net[id, i] = 1
    sim = np.dot(net, net.T)
    kout = np.sum(net.T, axis=0)
    oh = sparse.dia_matrix(([kout], [0]), shape=(N, N))
    matrix = oh.dot(np.ones((N,N)))
    denominator = np.minimum(matrix, matrix.T)
    for i in range(N):
        for j in range(N):
            if denominator[i, j] != 0:
                sim[i, j] = sim[i, j] / denominator[i, j]
            else:
                sim[i, j] = 0
    return sim

def calculate_cn_similarity(record, N):
    nn = record.shape[0]
    net = np.zeros([N, nn])
    for i in range(nn):
        id = record[i, np.where(record[i, :] > 0)]
        id = id.astype(int)
        net[id, i] = 1
    sim = np.dot(net, net.T)
    return sim

def calculate_jaccard_similarity(record, N):
    nn = record.shape[0]
    net = np.zeros([N, nn])
    for i in range(nn):
        id = record[i, np.where(record[i, :] > 0)]
        id = id.astype(int)
        net[id, i] = 1
    sim = np.dot(net, net.T)
    kout = np.sum(net.T, axis=0)
    oh = sparse.dia_matrix(([kout], [0]), shape=(N, N))
    matrix = oh.dot(np.ones((N,N)))
    denominator = matrix + (matrix.T) - sim
    for i in range(N):
        for j in range(N):
            if denominator[i, j] != 0:
                sim[i, j] = sim[i, j] / denominator[i, j]
            else:
                sim[i, j] = 0
    return sim

def calculate_HDI_similarity(record, N):
    nn = record.shape[0]
    net = np.zeros([N, nn])
    for i in range(nn):
        id = record[i, np.where(record[i, :] > 0)]
        id = id.astype(int)
        net[id, i] = 1
    sim = np.dot(net, net.T)
    kout = np.sum(net.T, axis=0)
    oh = sparse.dia_matrix(([kout], [0]), shape=(N, N))
    matrix = oh.dot(np.ones((N,N)))
    denominator = np.maximum(matrix, matrix.T)
    for i in range(N):
        for j in range(N):
            if denominator[i, j] != 0:
                sim[i, j] = sim[i, j] / denominator[i, j]
            else:
                sim[i, j] = 0
    return sim

def calculate_PA_similarity(record, N):
    nn = record.shape[0]
    net = np.zeros([N, nn])
    for i in range(nn):
        id = record[i, np.where(record[i, :] > 0)]
        id = id.astype(int)
        net[id, i] = 1
    kout = np.sum(net.T, axis=0)
    kout = kout.reshape(1, kout.shape[0])
    sim = np.dot(kout.T, kout)
    return sim

def calculate_LHN_similarity(record, N):
    nn = record.shape[0]
    net = np.zeros([N, nn])
    for i in range(nn):
        id = record[i, np.where(record[i, :] > 0)]
        id = id.astype(int)
        net[id, i] = 1
    sim = np.dot(net, net.T)
    kout = np.sum(net.T, axis=0)
    oh = sparse.dia_matrix(([kout], [0]), shape=(N, N))
    matrix = oh.dot(np.ones((N,N)))
    denominator = matrix * (matrix.T)
    for i in range(N):
        for j in range(N):
            if denominator[i, j] != 0:
                sim[i, j] = sim[i, j] / denominator[i, j]
            else:
                sim[i, j] = 0
    return sim

def calculate_RA_similarity(record, N):
    nn = record.shape[0]
    net = np.zeros([N, nn])
    for i in range(nn):
        id = record[i, np.where(record[i, :] > 0)]
        id = id.astype(int)
        net[id, i] = 1
    kitem = np.sum(net, axis=0)
    oh = sparse.dia_matrix(([kitem], [0]), shape=(N, N))
    matrix = oh.dot(np.ones((N, N)))
    net1 = net / (matrix.T)
    sim = np.dot(net, net1.T)
    asd2 = np.where(matrix == 0)
    sim[asd2] = 0
    return sim

def network_rebuild(sim, ratio, A):
    """
    参数:
        sim: 相似度矩阵
        ratio: 
        A: 网络的邻接矩阵
    
    返回值:
        B: 通过sim矩阵重构出的网络邻接矩阵


    """
    N = A.shape[0]
    r = np.random.rand(N, N)
    r = r + r.T
    sim = sim + 0.000001 * r
    sim = sim - np.diag(np.diag(sim))
    X = np.ceil(ratio * np.sum(A))
    n1 = sim[np.where(sim > 0)]
    n2 = -np.sort(-n1)
    threshold = n2[int(X)]
    B = np.zeros([N, N])
    B[np.where(sim >= threshold)] = 1
    return B

def calculate_AUC(A, sim):
    """
    参数:
        A: 网络的邻接矩阵
        sim: 相似度矩阵
    
    返回值:
        AUC: 返回AUC值


    """
    nn = 100000
    N = A.shape[1]
    mm = A.shape[0]
    if mm >= N:
        N = mm
    B = A
    B = B + 2 * np.eye(N, N)
    (l1, r1) = np.where(B == 1)
    (l0, r0) = np.where(B == 0)
    n1 = 0
    n2 = 0
    for i in range(nn):
        line1 = np.random.randint(len(l1))
        line0 = np.random.randint(len(l0))
        if sim[l1[line1], r1[line1]] > sim[l0[line0], r0[line0]]:
            n1 = n1 + 1
        if sim[l1[line1], r1[line1]] == sim[l0[line0], r0[line0]]:
            n2 = n2 + 1
    AUC = (n1 + n2 * 0.5) / nn
    return AUC

def Statistic(adjlist):
    degree = np.sum(adjlist, axis=0)
    degree = degree.astype(int)
    NoLink = 0
    nn = adjlist.shape[0]
    mm = adjlist.shape[1]
    lenth = mm
    if nn >= mm:
        lenth = nn
    cluster = np.zeros([1, lenth])[0]
    for i in range(lenth):
        temp = np.zeros([1, degree[i]])[0]
        temp = temp.astype(int)
        k = 0
        for j in range(lenth):
            if adjlist[i, j] > 0:
                temp[k] = j
                k = k + 1
        triangle = 0
        for j in range(degree[i] - 1):
            for k in range(j+1, degree[i]):
                if temp[k] != 0:
                    if (adjlist[temp[j], temp[k]] > 0 and adjlist[temp[j], temp[k]] != NoLink):
                        triangle = triangle + 1
        if (len(temp) > 1 and ((degree[i] - 1) * degree[i]) != 0):
            cluster[i] = 2 * triangle / ((degree[i] - 1) * degree[i])
    SNet = sparse.csr_matrix(adjlist)
    Graph = nx.from_scipy_sparse_matrix(SNet)
    shortpath_dict = dict(nx.all_pairs_shortest_path_length(Graph))
    shortpath = np.zeros([lenth, lenth])
    for i in range(lenth):
        shortpath[i, :] = -1
    for key1 in shortpath_dict:
        for key2 in shortpath_dict[key1]:
            shortpath[int(key1), int(key2)] = shortpath_dict[key1][key2]
    totalcluster = 0
    totalshortpath = 0
    totallink = 0
    for i in range(lenth):
        totalcluster = totalcluster + cluster[i]
        for j in range(lenth):
            if (shortpath[i, j] != -1 and shortpath[i, j] !=0):
                totalshortpath = totalshortpath + shortpath[i, j]
                totallink = totallink + 1
    avecluster = totalcluster / lenth
    aveshortpath = totalshortpath / lenth
    return degree, cluster, shortpath, avecluster, aveshortpath, totallink
