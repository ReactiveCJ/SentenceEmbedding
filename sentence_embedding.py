import numpy as np 
import torch

def global_metric(word_vecs):
    cov = np.cov(word_vecs.T)
    average = np.mean(word_vecs, axis=0)
    return np.linalg.inv(cov), average


def metric_distance(inverse_cov, vec1, vec2):
    return math.sqrt(np.matmul(vec1, inverse_cov).dot(vec2))

def metric_emb(word_vecs, sens, inverse_cov, global_avg, global_only):
    n_sen = len(sens)
    emb = np.empty((n_sen, word_vecs.shape[1]))
    for idx, sen in enumerate(sens):
        if len(sen) > 0:
            raw_emb = word_vecs[sen, :]
            if global_only:
                distance = np.array([metric_distance(inverse_cov, global_avg, vec) for vec in raw_emb])
            else:
                avg_emb = np.sum(raw_emb, axis=0) / len(sen)
                distance = np.array([metric_distance(inverse_cov, avg_emb, vec) for vec in raw_emb])
            avg_distance = distance/(2*np.average(distance))
            weights = np.array([1.8*(x - 0.5) + 0.5 for x in avg_distance])
            print(weights)
            emb[idx] = weights.dot(raw_emb)
        else:
            emb[idx] = np.zeros(word_vecs.shape[1])
    return torch.from_numpy(emb)

def average_emb(word_vecs, sens):
    n_sen = len(sens)
    emb = np.empty((n_sen, word_vecs.shape[1]))
    for idx, sen in enumerate(sens):
        if len(sen) > 0:
            emb[idx] = np.sum(word_vecs[sen, :], axis=0) / len(sen)
        else:
            emb[idx] = np.zeros(word_vecs.shape[1])
    return torch.from_numpy(emb)

def weighted_emb(word_vecs, sens, weights):
    n_sen = len(sens)
    emb = np.empty((n_sen, word_vecs.shape[1]), dtype=np.float32)
    for idx, sen in enumerate(sens):
        if len(sen) > 0:
            emb[idx] = weights[sen].dot(word_vecs[sen, :]) / len(sen)
        else:
            emb[idx] = np.zeros(word_vecs.shape[1])
    return torch.from_numpy(emb)
    
def first_pca(sen_vecs):
    U, S, V = torch.svd(sen_vecs)
    return torch.unsqueeze(V[0], 1)

def sif(sen_vecs, first_component):
    sif_embeddings = sen_vecs - sen_vecs.mm(first_component).mm(first_component.t())
    return sif_embeddings
