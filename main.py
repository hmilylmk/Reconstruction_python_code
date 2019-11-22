import numpy as np
import scipy.io as scio
from scipy import sparse
import math
from util import *
import json


class Reconstrution():
    def __init__(self, A, name, method="SIR", beta=0, virus=0, **kwargs):
        self.name = name
        self.A = A
        self.result_auc = {}
        self.result_pre = {}
        self.method = method
        self.beta = beta
        self.virus = virus
        self.record = None
        self.record_t = None
    
    def init(self):
        A = self.A
        N = A.shape[0]
        if (self.beta == 0):
            avek = np.sum(A) / N
            kd = np.sum(A, axis=0)
            kd = kd.A
            kd = kd ** 2
            kd = np.sum(kd) / N
            betac = avek / (kd - avek)
            beta = 4 * betac
            self.beta = beta
            if beta > 1:
                self.beta = 1
        if (self.virus == 0):
            self.virus = math.ceil(0.5 * N)
    
    def spreading(self):
        method = self.method
        virus = self.virus
        record = np.zeros((virus, N))
        record_t = np.zeros((virus, N))
        beta = self.beta
        if method == "SIR":
            for i in range(virus):
                origin = np.random.randint(0, N)
                coverage, speed, t = SIR(A, origin, beta)
                record[i, :] = -1
                record_t[i, :] = -1
                record[i, :len(coverage)] = coverage
                record_t[i, :len(coverage)] = t
        if method == "SI":
            for i in range(virus):
                origin = np.random.randint(0, N)
                coverage, speed, t = SI(A, origin, beta)
                record[i, :] = -1
                record_t[i, :] = -1
                record[i, :len(coverage)] = coverage
                record_t[i, :len(coverage)] = t
        if method == "LT":
            for i in range(virus):
                origin = np.random.randint(0, N)
                coverage, speed, t = LT(A, origin, beta)
                record[i, :] = -1
                record_t[i, :] = -1
                record[i, :len(coverage)] = coverage
                record_t[i, :len(coverage)] = t
        
        self.record = record
        self.record_t = record_t
    
    def reconstrut(self, times):
        record = self.record
        record_t = self.record_t
        A = self.A
        nn = record.shape[0]
        mm = record.shape[1]
        tnet = np.zeros([N, nn])
        net = np.zeros([N, nn])
        for i in range(nn):
            id = record[i, np.where(record[i,:] >= 0)]
            id = id.astype(int)
            net[id, i] = 1
            tnet[id, i] = record_t[i, np.where(record_t[i,:] >= 0)]
        kout = np.sum(net.T, axis=0)
        knews = np.sum(net, axis=0)
        pnet = np.dot(net, net.T)
        oh = sparse.dia_matrix(([kout], [0]), shape=(N,N))
        matrix = oh.dot(np.ones((N,N)))
        # cos denominator
        denominator = (matrix * (matrix.T)) ** 0.5
        # jaccard denominator
        denominator_jaccard = matrix + (matrix.T) - pnet
        # lhn denominator
        denominator_lhn = matrix * (matrix.T)
        # ssi denominator
        denominator_ssi = matrix + (matrix.T)
        # hdi denominator
        denominator_hdi = np.maximum(matrix, matrix.T)
        # hpi denominator
        denominator_hpi = np.minimum(matrix, matrix.T)
        # pa denominator
        denominator_pa = matrix * (matrix.T)


        sim_cos_t = np.zeros([N,N])
        sim_ssi_t = np.zeros([N,N])
        sim_hpi_t = np.zeros([N,N])
        sim_cn_t = np.zeros([N,N])
        sim_jard_t = np.zeros([N,N])
        sim_hdi_t = np.zeros([N,N])
        sim_pa_t = np.zeros([N,N])
        sim_lhn_t = np.zeros([N,N])
        sim_ra_t = np.zeros([N,N])

        sim_cos_t_1 = np.zeros([N,N])
        sim_ssi_t_1 = np.zeros([N,N])
        sim_hpi_t_1 = np.zeros([N,N])
        sim_cn_t_1 = np.zeros([N,N])
        sim_jard_t_1 = np.zeros([N,N])
        sim_hdi_t_1 = np.zeros([N,N])
        sim_pa_t_1 = np.zeros([N,N])
        sim_lhn_t_1 = np.zeros([N,N])
        sim_ra_t_1 = np.zeros([N,N])

        for i in range(N):
            for j in range(N):
                if i != j:
                    id1 = np.where(tnet[i, :] > 0)
                    id2 = np.where(tnet[j, :] > 0)
                    id = np.intersect1d(id1, id2)
                    id = id.astype(int)
                    p = []
                    if len(id) > 0:
                        kn = knews[id]
                        p = np.abs(tnet[i, id] - tnet[j, id])
                        idp1 = np.where(p == 1)
                        p_zero = np.where(p == 0)
                        if len(p_zero) > 0:
                            p[p_zero] = 100
                        kn1 = knews[idp1]
                        p1 = p[idp1]
                        sim_cn_t_1[i, j] = np.sum(1 / p1)
                        sim_ra_t_1[i, j] = np.sum(1 / (kn1*p1))
                        sim_cn_t[i, j] = np.sum(1 / p)
                        sim_ra_t[i, j] = np.sum(1 / (kn * p))
                        if denominator[i, j] != 0:
                            sim_cos_t[i, j] = np.sum(1 / p) / denominator[i, j]
                            sim_ssi_t[i, j] = np.sum(1 / p) / denominator_ssi[i, j]
                            sim_hpi_t[i, j] = np.sum(1 / p) / denominator_hpi[i, j]
                            sim_jard_t[i, j] = np.sum(1 / p) / denominator_jaccard[i, j]
                            sim_hdi_t[i, j] = np.sum(1 / p) / denominator_hdi[i, j]
                            sim_pa_t[i, j] = np.sum(1 / p) * denominator_pa[i, j]
                            sim_cos_t_1[i, j] = np.sum(1 / p1) / denominator[i, j]
                            sim_ssi_t_1[i, j] = np.sum(1 / p1) / denominator_ssi[i, j]
                            sim_hpi_t_1[i, j] = np.sum(1 / p1) / denominator_hpi[i, j]
                            sim_jard_t_1[i, j] = np.sum(1 / p1) / denominator_jaccard[i, j]
                            sim_hdi_t_1[i, j] = np.sum(1 / p1) / denominator_hdi[i, j]
                            sim_pa_t_1[i, j] = np.sum(1 / p1) * denominator_pa[i, j]
                        if denominator_lhn[i, j] != 0:
                            sim_lhn_t[i, j] = np.sum(1 / p) / denominator_lhn[i, j]
                            sim_lhn_t_1[i, j] = np.sum(1 / p1) / denominator_lhn[i, j]

        ratio = 1
        sim_cos = calculate_salton_similarity(record, N)
        B = network_rebuild(sim_cos, ratio, A)
        Pre_cos = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_cos_t, ratio, A)
        Pre_cos_t = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_cos_t_1, ratio, A)
        Pre_cos_t_1 = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))

        sim_ssi = calculate_Sorensen_similarity(record, N)
        B = network_rebuild(sim_ssi, ratio, A)
        Pre_ssi = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_ssi_t, ratio, A)
        Pre_ssi_t = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_ssi_t_1, ratio, A)
        Pre_ssi_t_1 = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))

        sim_hpi = calculate_HPI_similarity(record, N)
        B = network_rebuild(sim_hpi, ratio, A)
        Pre_hpi = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_hpi_t, ratio, A)
        Pre_hpi_t = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_hpi_t_1, ratio, A)
        Pre_hpi_t_1 = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))

        sim_cn = calculate_cn_similarity(record, N)
        B = network_rebuild(sim_cn, ratio, A)
        Pre_cn = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_cn_t, ratio, A)
        Pre_cn_t = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_cn_t_1, ratio, A)
        Pre_cn_t_1 = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))

        sim_jard = calculate_jaccard_similarity(record, N)
        B = network_rebuild(sim_jard, ratio, A)
        Pre_jard = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_jard_t, ratio, A)
        Pre_jard_t = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_jard_t_1, ratio, A)
        Pre_jard_t_1 = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))

        sim_hdi = calculate_HDI_similarity(record, N)
        B = network_rebuild(sim_hdi, ratio, A)
        Pre_hdi = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_hdi_t, ratio, A)
        Pre_hdi_t = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_hdi_t_1, ratio, A)
        Pre_hdi_t_1 = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))

        sim_pa = calculate_PA_similarity(record, N)
        B = network_rebuild(sim_pa, ratio, A)
        Pre_pa = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_pa_t, ratio, A)
        Pre_pa_t = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_pa_t_1, ratio, A)
        Pre_pa_t_1 = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))

        sim_lhn = calculate_LHN_similarity(record, N)
        B = network_rebuild(sim_lhn, ratio, A)
        Pre_lhn = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_lhn_t, ratio, A)
        Pre_lhn_t = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))
        B = network_rebuild(sim_lhn_t_1, ratio, A)
        Pre_lhn_t_1 = np.sum(np.sum(np.multiply(A, B), axis=0)) / np.sum(np.sum(A, axis=0))

        


        AUC_cos = calculate_AUC(A, sim_cos) 
        AUC_cos_t = calculate_AUC(A, sim_cos_t) 
        AUC_cos_t_1 = calculate_AUC(A, sim_cos_t_1) 
        AUC_ssi = calculate_AUC(A, sim_ssi) 
        AUC_ssi_t = calculate_AUC(A, sim_ssi_t) 
        AUC_ssi_t_1 = calculate_AUC(A, sim_ssi_t_1) 
        AUC_hpi = calculate_AUC(A, sim_hpi) 
        AUC_hpi_t = calculate_AUC(A, sim_hpi_t) 
        AUC_hpi_t_1 = calculate_AUC(A, sim_hpi_t_1) 
        AUC_cn = calculate_AUC(A, sim_cn) 
        AUC_cn_t = calculate_AUC(A, sim_cn_t) 
        AUC_cn_t_1 = calculate_AUC(A, sim_cn_t_1) 
        AUC_jard = calculate_AUC(A, sim_jard) 
        AUC_jard_t = calculate_AUC(A, sim_jard_t) 
        AUC_jard_t_1 = calculate_AUC(A, sim_jard_t_1) 
        AUC_hdi = calculate_AUC(A, sim_hdi) 
        AUC_hdi_t = calculate_AUC(A, sim_hdi_t) 
        AUC_hdi_t_1 = calculate_AUC(A, sim_hdi_t_1) 
        AUC_pa = calculate_AUC(A, sim_pa) 
        AUC_pa_t = calculate_AUC(A, sim_pa_t) 
        AUC_pa_t_1 = calculate_AUC(A, sim_pa_t_1) 
        AUC_lhn = calculate_AUC(A, sim_lhn) 
        AUC_lhn_t = calculate_AUC(A, sim_lhn_t) 
        AUC_lhn_t_1 = calculate_AUC(A, sim_lhn_t_1) 

        result_auc = {"AUC_cos": AUC_cos, 
                    "AUC_cos_t": AUC_cos_t,
                    "AUC_cos_t_1": AUC_cos_t_1,
                    "AUC_ssi": AUC_ssi, 
                    "AUC_ssi_t": AUC_ssi_t,
                    "AUC_ssi_t_1": AUC_ssi_t_1,
                    "AUC_hpi": AUC_hpi, 
                    "AUC_hpi_t": AUC_hpi_t,
                    "AUC_hpi_t_1": AUC_hpi_t_1,
                    "AUC_cn": AUC_cn, 
                    "AUC_cn_t": AUC_cn_t,
                    "AUC_cn_t_1": AUC_cn_t_1,
                    "AUC_jard": AUC_jard, 
                    "AUC_jard_t": AUC_jard_t,
                    "AUC_jard_t_1": AUC_jard_t_1,
                    "AUC_hdi": AUC_hdi, 
                    "AUC_hdi_t": AUC_hdi_t,
                    "AUC_hdi_t_1": AUC_hdi_t_1,
                    "AUC_pa": AUC_pa, 
                    "AUC_pa_t": AUC_pa_t,
                    "AUC_pa_t_1": AUC_pa_t_1,
                    "AUC_lhn": AUC_lhn, 
                    "AUC_lhn_t": AUC_lhn_t,
                    "AUC_lhn_t_1": AUC_lhn_t_1}
        result_pre = {"Pre_cos": Pre_cos, 
                    "Pre_cos_t": Pre_cos_t,
                    "Pre_cos_t_1": Pre_cos_t_1,
                    "Pre_ssi": Pre_ssi, 
                    "Pre_ssi_t": Pre_ssi_t,
                    "Pre_ssi_t_1": Pre_ssi_t_1,
                    "Pre_hpi": Pre_hpi, 
                    "Pre_hpi_t": Pre_hpi_t,
                    "Pre_hpi_t_1": Pre_hpi_t_1,
                    "Pre_cn": Pre_cn, 
                    "Pre_cn_t": Pre_cn_t,
                    "Pre_cn_t_1": Pre_cn_t_1,
                    "Pre_jard": Pre_jard, 
                    "Pre_jard_t": Pre_jard_t,
                    "Pre_jard_t_1": Pre_jard_t_1,
                    "Pre_hdi": Pre_hdi, 
                    "Pre_hdi_t": Pre_hdi_t,
                    "Pre_hdi_t_1": Pre_hdi_t_1,
                    "Pre_pa": Pre_pa, 
                    "Pre_pa_t": Pre_pa_t,
                    "Pre_pa_t_1": Pre_pa_t_1,
                    "Pre_lhn": Pre_lhn, 
                    "Pre_lhn_t": Pre_lhn_t,
                    "Pre_lhn_t_1": Pre_lhn_t_1}
        print(result_auc)
        print(result_pre)
        fwrite_auc = open("./{}_{}_result_auc_{}.json".format(self.name, self.method, times), "w")
        fwrite_pre = open("./{}_{}_result_pre_{}.json".format(self.name, self.method, times), "w")
        json.dump(result_auc, fwrite_auc)
        json.dump(result_pre, fwrite_pre)


    def run(self, times=1):
        self.init()
        for i in range(times):
            self.spreading()
            self.reconstrut(i)


if __name__ == "__main__":

    # 导入数据，保证A为网络邻接矩阵即可
    data_path = "./word(1).mat"
    data = scio.loadmat(data_path)
    A = data['A']
    A = A.todense()
    N = A.shape[0]
    rt = Reconstrution(A=A, name="word", method="LT")
    rt.run(5)
    
    

