def global_metric(word_vec):
    cov = np.cov(word_vec.T)
    average = np.mean(word_vec, axis=0)
    return np.linalg.inv(cov), average


def metric_distance(inverse_cov, vec1, vec2):
    return math.sqrt(np.matmul(vec1, inverse_cov).dot(vec2))

def metric_emb(wv, sens, inverse_cov, global_avg, global_only):
    n_sen = len(sens)
    emb = np.empty((n_sen, wv.shape[1]))
    for idx, sen in enumerate(sens):
        if len(sen) > 0:
            raw_emb = wv[sen, :]
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
            emb[idx] = np.zeros(wv.shape[1])

    return torch.from_numpy(emb)

def average_emb(wv, sens):
    n_sen = len(sens)
    emb = np.empty((n_sen, wv.shape[1]))
    for idx, sen in enumerate(sens):
        if len(sen) > 0:
            emb[idx] = np.sum(wv[sen, :], axis=0) / len(sen)
        else:
            emb[idx] = np.zeros(wv.shape[1])
    return torch.from_numpy(emb)

def weighted_emb(wv, sens, weights):
    n_sen = len(sens)
    emb = np.empty((n_sen, wv.shape[1]), dtype=np.float32)
    for idx, sen in enumerate(sens):
        if len(sen) > 0:
            emb[idx] = weights[sen].dot(wv[sen, :]) / len(sen)
        else:
            emb[idx] = np.zeros(wv.shape[1])
    return torch.from_numpy(emb)
    
def first_pca(sen_vecs):
    U, S, V = torch.svd(sen_vecs)
    return torch.unsqueeze(V[0], 1)

def sif(sen_vecs, first_component):
    sif_embeddings = sen_vecs - sen_vecs.mm(first_component).mm(first_component.t())
    return sif_embeddings
