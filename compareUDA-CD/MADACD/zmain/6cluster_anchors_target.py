import os

import numpy as np
import torch
import torch.nn.parallel

from utils.clustering import run_kmeans

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    seed = 31
    # fix random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # load and transition to 24966*(19*256)
    CAU = torch.load('./output/G-L/Target_objective_vectors.pkl')
    x = np.reshape(CAU, (CAU.shape[0], CAU.shape[1] * CAU.shape[2])).astype('float32')

    # kmeans
    ncentroids = 10
    cluster_centroids, cluster_index, cluster_loss = run_kmeans(x, ncentroids, verbose=True)

    '''
    # origin cluster
    ncentroids = 10
    niter = 20
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=True, gpu=True)
    kmeans.train(x)
    # get the result
    cluster_result = kmeans.centroids
    cluster_loss = kmeans.obj
    '''
    print(cluster_centroids)
    print(len(cluster_index))#6628
    print(cluster_loss)#36002.3828125
    torch.save(cluster_centroids, './output/G-L/anchors/Target_cluster_centroids_full_%d.pkl' % ncentroids)
    torch.save(cluster_index, './output/G-L/anchors/Target_cluster_index_full_%d.pkl' % ncentroids)
    '''
    # import cluster
    nmb = 10
    deepcluster = clustering.Kmeans(nmb)
    clustering_loss = deepcluster.cluster(CAU, verbose=True)
    '''


if __name__ == '__main__':
    main()
