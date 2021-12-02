import numpy as np
import matplotlib.pyplot as plt

def perform_MDA(dataset, flattened, subjects, types):
    noise = {'pose':0.93, 'face':0.999}

    print('Performing MDA, please wait....')

    mu_class = [] # emperical mean of each class
    cov_class = [] # emperical covariace of each class

    mu_not = np.zeros(shape=(flattened.shape[1]))
    sigma_b = np.zeros(shape=(flattened.shape[1], flattened.shape[1]))
    sigma_w = np.zeros(shape=(flattened.shape[1], flattened.shape[1]))
    prob = 1/subjects

    for i in range(subjects):
        # determine class as per the data
        start = i*types
        end = (i+1)*types
        
        temp = flattened[start:end]
        mean = np.mean(temp, axis=0)
        mu_class.append(mean)

        cov = np.zeros(shape=(temp.shape[1], temp.shape[1]))
        for k in range(types):
            mat = (temp[k] - mean).reshape(temp.shape[1], 1)
            cov = cov + np.dot(mat, mat.T)
        # mean of covariance
        cov = cov / types
        # add noise
        cov += noise.get(dataset, 0.8)*np.identity(cov.shape[0])
        cov_class.append(cov)
        # print(np.linalg.det(cov))
        # break

        # emperical mean mu_not
        mu_not += prob*mean

        #sigma_w
        sigma_w += prob*cov

    # #calculate the sigma_b
    for i in range(subjects):
        mat = (mu_class[i] - mu_not).reshape(flattened.shape[1], 1)
        sigma_b += prob*np.dot(mat, mat.T)

    # # pose   
    # b = sigma_b + 0.747*np.identity(cov.shape[0])   #0.745999
    # np.linalg.det(b)

    # # face
    b = sigma_b + 0.945*np.identity(cov.shape[0])
    np.linalg.det(b)

    # a = sigma_w #+ np.identity(cov_class[0].shape[0])
    a = np.dot(np.linalg.inv(sigma_w), b)

    # find the eigen values and eigen vectors
    val, vec = np.linalg.eig(a)

    # sort the eigen values and corresponding vectors
    idx = val.argsort()[::-1]

    val_ = val[idx]
    vec_  = vec[:, idx]
    
    # take the maximum possible dimensions
    dim = 60#subjects - 1
    # final vectors till the dimension
    final_vectors = vec_[:,:dim]

    # get the projection
    projection = np.dot(flattened, final_vectors)

    print(final_vectors)
    print('-----------------------------------------')
    print(val[:dim])
    print('-----------------------------------------')

    return projection