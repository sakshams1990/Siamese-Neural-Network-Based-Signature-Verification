from scipy.spatial import distance


def cosine_similarity_score(i1, i2):
    cos_sim = 1 - distance.cosine(i1, i2)
    return cos_sim
