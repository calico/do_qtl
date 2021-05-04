import numpy as np

#pylint: disable-msg=invalid-name

def outer_product(A, B):
    """Compute an outer product of two matrices.
    """
    return np.array([a*b for a in A for b in B])

def jackknife(N, n, K, X=None, blocks=None):
    """
    Split N samples into n test and N-n train, k times.
    Ensure each split allows all columns of X to be non-zero
    and all sub-parts of blocks to be non-zero, for both test
    and train.
    """

    jackknifed_samples = []
    k = 0
    while k < K:
        all_samples = np.arange(N)
        test_samples = np.random.choice(np.arange(N), size=n, replace=False)
        train_samples = np.delete(all_samples, test_samples)

        # check if all columns of train / test split are non-zero
        if X is not None:
            if np.any(np.sum(X[train_samples], 0) == 0):
                continue

        # check if train/test sub-blocks of all blocks are non-zero
        if blocks is not None:
            if np.any([np.sum(block[train_samples, :][:, train_samples]) == 0 for block in blocks]):
                continue

        jackknifed_samples.append(test_samples)
        k += 1

    return jackknifed_samples
