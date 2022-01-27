import numpy as np
from sklearn.manifold import MDS
from scipy.spatial import distance_matrix
import scipy.linalg
import scipy.ndimage.measurements
import matplotlib.pyplot as plt

'''
### USING NON-ITERATIVE MDS ###

MATLAB Code:
n = size(D,1);
J = eye(n) - ones( n)/n;
B = -(1/2) * J*D*J;
[u,s,v] = svd(B);
rows = 1:ndim; // p in the paper
Xout = u(:,rows)*s(rows,rows)^(1/2);

'''
def classical(D, p):
    '''
    mds.classical applies non-iterative classical multi-dimensional scaling algorithm

    :param D: squared-distance matrix between points
    :param p: dimension count for MDS algorithm
    :return: MDS algorithm output
    '''
    n = D.shape[0]
    J = np.eye(n) - np.ones(n)/n
    B = -(1/2) * J @ D @ J
    u, s, v = np.linalg.svd(B)
    X_out = u[:, :p] @ scipy.linalg.sqrtm(np.diag(s)[:p, :p])
    return X_out


'''
### RMS for MDS ###
Y: ground truth matrix with yi.T is ith row
X: MDS estimation matrix with xi.T is ith row

A = (X' * Y * Y' * X)^(1/2) * (Y' * X)^(-1)

cum_error = 0
for i = 1:n
    error = A' * x[i, :] - y[i, :]
    cum_error += norm(error)^2
rms = sqrt(cum_error / n)
'''
def rms(X, Y):
    '''
    mds.rms calculates RMS value for MDS outputs by best aligning the points by rotating and reflecting them.

    :param X: MDS estimation matrix
    :param Y: ground truth matrix
    '''
    X_ = np.copy(X)
    Y_ = np.copy(Y)

    X_ -= X_.mean(axis=0)
    Y_ -= Y_.mean(axis=0)

    n = X_.shape[0]
    assert(X_.shape == Y_.shape)

    A = scipy.linalg.sqrtm(X_.T @ Y_ @ Y_.T @ X_) @ scipy.linalg.inv(Y_.T @ X_)

    cum_error = 0
    for i in range(n):
        x_vec, y_vec = X_[i, :].T, Y_[i, :].T
        error = A.T @ x_vec - y_vec
        assert (1 in error.shape) or (len(error.shape) == 1), f"Internal error is not vector: {error.shape}" # Check if vector
        cum_error += np.linalg.norm(error) ** 2

    rms = np.sqrt(cum_error / n)
    return rms


def plot_2d(X, ax=None, remove_min=False):
    '''
    mds.plot_2d

    :param X: position matrix (row vectors)
    '''
    assert X.shape[1] == 2

    if ax is None:
        ax = plt.gca()

    min_x, min_y = X.min(axis=0)
    centroid_X = X.mean(axis=0)

    if remove_min:
        ax.scatter(X[:, 0] - min_x, X[:, 1] - min_y)
        ax.scatter(centroid_X[0] - min_x, centroid_X[1] - min_y, color='r', marker='x')
    else:
        ax.scatter(X[:, 0], X[:, 1])
        ax.scatter(centroid_X[0], centroid_X[1], color='r', marker='x')

    ax.set(xlabel='x', ylabel='y')
    ax.legend(["Points", "Centroid"])

    return ax
    

def classical_from_points(Y, p, verbose=True):
    '''
    :param Y: position matrix (row vectors)
    :param p: # of dimensions
    '''
    distMtx = distance_matrix(Y, Y)
    D = distMtx ** 2
    X = classical(D, p)
    return X


def basis_point(D, p, chosen_indices=None):
    '''
    D: pairwise distance matrix
    p: # of dimensions
    chosen_indices: Chosen point indices corresponding to that of the pairwise distance matrix
        if None, indices chosen randomly
    '''
    assert p == 3 # TODO
    n = D.shape[0]
    assert n == D.shape[1] 

    # Choose p+1 basis points
    if chosen_indices:
        assert len(chosen_indices) == p+1
        # D_basis = D[chosen_indices]
        bi = chosen_indices
    else:
        _random_indices = np.random.choice(n, p+1, replace=False)
        print("Indices chosen for basis-points:", _random_indices)
        # D_basis = D[_random_indices]
        bi = _random_indices

    '''
    Part 1
    ------
    Compute coordinates of the four basis points from their pairwise distances  
    '''
    # bp_A
    # xA = -(1/2) * D_basis[0, 1]
    xA = -(1/2) * D[bi[0], bi[1]]
    yA = 0
    zA = 0
    bp_A = np.array([xA, yA, zA])

    # bp_B
    # xB = (1/2) * D_basis[0, 1]
    xB = (1/2) * D[bi[0], bi[1]]
    yB = 0
    zB = 0
    bp_B = np.array([xB, yB, zB])

    def xQ(i):
        return ( D[bi[0], i]**2 - D[bi[1], i]**2 ) / (2*D[bi[0], bi[1]])

    # bp_C
    xC = xQ(bi[2]) # 2 for C
    yC = np.sqrt(
        (1/2)*D[bi[0],bi[2]]**2 - (1/4)*D[bi[0],bi[1]]**2 + (1/2)*D[bi[1],bi[2]]**2 - xC**2
    )
    zC = 0
    bp_C = np.array([xC, yC, zC])

    def yQ(i):
        return (D[bi[0], bi[2]]**2 - D[bi[0], bi[1]]**2 + D[bi[1], bi[2]]**2 + D[bi[0], i]**2 + D[bi[1], i]**2 - 2*D[bi[2], i]**2 - 4*xC*xQ(i)) / (4*yC)

    # bp_D
    xD = xQ(bi[3]) # 3 for D
    yD = yQ(bi[3])
    zD = np.sqrt((1/2)*D[bi[0], bi[3]]**2 - (1/4)*D[bi[0], bi[1]]**2 + (1/2)*D[bi[1], bi[3]]**2 - xD**2 - yD**2)
    bp_D = np.array([xD, yD, zD])

    '''
    Part 2
    ------
    Compute the coordinates of each microphone Q given its distance to the basis points 
    '''

    def zQ(i):
        return (1/(4*zD)) * (D[bi[0], bi[3]]**2 + D[bi[1], bi[3]]**2 + D[bi[0], i]**2 + D[bi[1], i]**2 - D[bi[0], bi[1]]**2 - 2*D[bi[3], i]**2 - 4*xD*xQ(i) - 4*yD*yQ(i))

    X = np.zeros((n, p))
    for i in range(n):
        X[i, :] = [xQ(i), yQ(i), zQ(i)]

    '''
    Part 3
    Build the squared-distance matrix D using these coordinates, and run the classical MDS algorithm
    '''

    return classical(distance_matrix(X, X) ** 2, p), bi


def basis_point_from_points(Y, p):
    '''
    :param Y: position matrix (row vectors)
    :param p: # of dimensions
    '''
    D = distance_matrix(Y, Y)
    X = basis_point(D, p)
    return X


def plot_3d(X, chosen_indices, ax, remove_min=False):
    '''
    mds.plot_3d

    :param X: position matrix (row vectors)
    '''
    assert X.shape[1] == 3
    assert len(chosen_indices) == 4

    min_x, min_y, min_z = X.min(axis=0)
    centroid_X = X.mean(axis=0)

    basisPts = X[chosen_indices, :]
    otherPts = np.delete(X, chosen_indices, axis=0)

    if remove_min:
        ax.scatter3D(otherPts[:, 0] - min_x, X[:, 1] - min_y, X[:, 2] - min_z)
        ax.scatter3D(basisPts[:, 0] - min_x, basisPts[:, 1] - min_y, basisPts[:, 2] - min_z)
        ax.scatter3D(centroid_X[0] - min_x, centroid_X[1] - min_y, centroid_X[2] - min_z, color='r', marker='x')
    else:
        ax.scatter3D(otherPts[:, 0], otherPts[:, 1], otherPts[:, 2])
        ax.scatter3D(basisPts[:, 0], basisPts[:, 1], basisPts[:, 2], color='g', marker='*')
        ax.scatter3D(centroid_X[0], centroid_X[1], centroid_X[2], color='r', marker='x')

    return ax