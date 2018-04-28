def average_emb(word_vecs, sens):
    """
    Compute the sentence weighted average vectors
    :param word_vecs: wv[i,:] - vector of word i - V*D
    :param sens: sens[i] - word indices list of sentence i - N * Unknown
    :return: emb[i, :] - embedding of sentence i - N * V
    """
    n_sen = len(sens)
    emb = np.empty((n_sen, word_vecs.shape[1]))
    for idx, sen in enumerate(sens):
        if len(sen) > 0:
            emb[idx] = np.sum(word_vecs[sen, :], axis=0) / len(sen)
        else:
            emb[idx] = np.zeros(word_vecs.shape[1])
    return emb


def weighted_emb(word_vecs, sens, weights):
    """
    Compute the sentence weighted average vectors
    :param word_vecs: wv[i,:] - vector of word i - V*D
    :param sens: sens[i] - word indices list of sentence i - N * Unknown
    :param weights: weights[i] - weight of word i - V * 1
    :return: emb[i, :] - embedding of sentence i - N * V
    """
    n_sen = len(sens)
    emb = np.empty((n_sen, word_vecs.shape[1]), dtype=np.float32)
    for idx, sen in enumerate(sens):
        if len(sen) > 0:
            emb[idx] = weights[sen].dot(word_vecs[sen, :]) / len(sen)
        else:
            emb[idx] = np.zeros(word_vecs.shape[1])
    return emb


def first_pca(sen_vecs):
    """
    :param sen_vecs: sentence embeddings, no need normalization 
    :return: first component computed from pca
    """
    U, S, V = np.linalg.svd(sen_vecs)
    return V[:, 0:1]


def sif_emb(sen_vecs, first_component):
    """
    first component
    :param sen_vecs: sentence embeddings  - N * V
    :param first_component: compute by pca - Ｖ * 1
    :return: new sentence embeddings - N * V
    """
    emb = sen_vecs - (sen_vecs.dot(first_component)).dot(first_component.T)
    return emb


def global_metric(word_vec):
    cov = np.cov(word_vec.T)
    average = np.mean(word_vec, axis=0)
    return np.linalg.inv(cov), average


def metric_distance(inverse_cov, vec1, vec2):
    return math.sqrt(np.matmul(vec1, inverse_cov).dot(vec2))


def metric_embedding(word_vecs, sens, inverse_cov, global_avg, global_only):
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
            emb[idx] = weights.dot(raw_emb)
        else:
            emb[idx] = np.zeros(word_vecs.shape[1])

    return emb


wvs = np.random.randn(20, 5)
sens = [0, 4, 6, 11]
sens_vec = average_emb(wvs, sens)
