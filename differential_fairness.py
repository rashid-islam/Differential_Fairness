import numpy as np

# Computing Differential Fairness metric using smoothed EDF method (Equation-6).
# Source: James R. Foulds, Rashidul Islam, Kamrun Naher Keya, and Shimei Pan. An Intersectional Definition of Fairness. ArXiv preprint arXiv:1807.08362 [CS.LG], 2018
# Link: https://arxiv.org/pdf/1807.08362.pdf

#%%
# Measure intersectional DF from positive predict probabilities
def differentialFairnessBinaryOutcome(probabilitiesOfPositive):
    # input: probabilitiesOfPositive = positive p(y=1|S) from data or ML algorithm
    # output: overall differential fairness measure (epsilon)
    epsilonPerGroup = np.zeros(len(probabilitiesOfPositive))
    for i in  range(len(probabilitiesOfPositive)):
        epsilon = 0.0 # initialization of DF
        for j in range(len(probabilitiesOfPositive)):
            if i == j:
                continue
            else:
                epsilon = max(epsilon,abs(np.log(probabilitiesOfPositive[i])-np.log(probabilitiesOfPositive[j]))) # ratio of probabilities of positive outcome
                epsilon = max(epsilon,abs(np.log((1-probabilitiesOfPositive[i]))-np.log((1-probabilitiesOfPositive[j])))) # ratio of probabilities of negative outcome
        epsilonPerGroup[i] = epsilon # DF per group
    epsilon = max(epsilonPerGroup) # overall DF of the mechanism 
    return epsilon

#%%
# smoothed empirical differential fairness (EDF) measurement
def computeSmoothedEDF(protectedAttributes,predictions):
    # Dirichlet smoothing parameters
    numClasses = 2
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
        
    # compute counts for each intersectional group
    intersectGroups = np.unique(protectedAttributes,axis=0) # all intersecting groups, i.e. black-women, white-man etc
    countsClassOne = np.zeros((len(intersectGroups)))
    countsTotal = np.zeros((len(intersectGroups)))
    for i in range(len(predictions)):
        index=np.where((intersectGroups==protectedAttributes[i]).all(axis=1))[0][0]
        countsTotal[index] += 1
        if predictions[i] == 1:
            countsClassOne[index] += 1
    
    # probability of y given S (p(y=1|S))
    probabilitiesForDFSmoothed = (countsClassOne + dirichletAlpha) /(countsTotal + concentrationParameter)

    # smoothed empirical differential fairness          
    epsilonSmoothed = differentialFairnessBinaryOutcome(probabilitiesForDFSmoothed)
    return epsilonSmoothed