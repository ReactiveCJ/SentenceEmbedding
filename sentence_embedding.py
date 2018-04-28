import numpy as np

def average_embedding(word_vecs, sens):
    """
    Compute the sentence weighted average embedding vectors
    :param word_vecs: word_vecs[i,:] - vector of word i - V * D
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


def weighted_embedding(word_vecs, sens, weights):
    """
    Compute the sentence weighted average embedding vectors
    :param word_vecs: word_vecs[i,:] - vector of word i - V * D
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
    :param sen_vecs: sentence embedding vectors, no need for normalization
    :return: first component computed from pca
    """
    U, S, V = np.linalg.svd(sen_vecs)
    return V[:, 0:1]


def sif_embedding(sen_vecs, first_component):
    """
    first component
    :param sen_vecs: sentence embedding vectors -  N * V
    :param first_component: compute by pca - V * 1
    :return: new sentence embedding vecs - N * V
    """
    emb = sen_vecs - (sen_vecs.dot(first_component)).dot(first_component.T)
    return emb


def global_metric(word_vecs):
    cov = np.cov(word_vecs.T)
    average = np.mean(word_vecs, axis=0)
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
sentence_list = [[0, 4, 6, 11]]
sens_vec = average_embedding(wvs, sentence_list)
