'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import scipy.ndimage as sp
import numpy as np
import pylab
import sys
import argparse
import os
        
def sparsenMatrix(baseMatrix, pConn):
    weightMatrix = np.zeros(baseMatrix.shape)
    numWeights = 0
    numTargetWeights = baseMatrix.shape[0] * baseMatrix.shape[1] * pConn
    weightList = [0]*int(numTargetWeights)
    while numWeights < numTargetWeights:
        idx = (np.int32(np.random.rand()*baseMatrix.shape[0]), np.int32(np.random.rand()*baseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = baseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], baseMatrix[idx])
            numWeights += 1
    return weightMatrix, weightList
        
    
def create_weights(label):    
    nInput = 784
    nE = 400
    nI = nE 
    dataPath = './simulations/'+label+'/random/'
    weight = {}
    weight['ee_input'] = 0.3 
    weight['ei_input'] = 0.2 
    weight['ee'] = 0.1
    weight['ei'] = 10.4
    weight['ie'] = 17.0
    weight['ii'] = 0.4
    pConn = {}
    pConn['ee_input'] = 1.0 
    pConn['ei_input'] = 0.1 
    pConn['ee'] = 1.0
    pConn['ei'] = 0.0025
    pConn['ie'] = 0.9
    pConn['ii'] = 0.1
    
    
    print('Creating random connection matrix between Xe and Ae...')
    connNameList = ['XeAe']
    for name in connNameList:
        weightMatrix = np.random.random((nInput, nE)) + 0.01
        weightMatrix *= weight['ee_input']
        if pConn['ee_input'] < 1.0:
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee_input'])
        else:
            weightList = [(i, j, weightMatrix[i,j]) for j in range(nE) for i in range(nInput)]
        np.save(dataPath+name, weightList)

    print('Creating connection matrices from Ae->Ai which are constant... ')
    connNameList = ['AeAi']
    for name in connNameList:
        weightList = [(i, i, weight['ei']) for i in range(nE)]
        np.save(dataPath+name, weightList)
        
    print('Creating connection matrices from Ai->Ae which are constant...')
    connNameList = ['AiAe']
    for name in connNameList:
        weightMatrix = np.ones((nI, nE))
        weightMatrix *= weight['ie']
        for i in range(nI):
            weightMatrix[i,i] = 0
        weightList = [(i, j, weightMatrix[i,j]) for i in range(nI) for j in range(nE)]
        np.save(dataPath+name, weightList)

         
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Script to initialize the directory structure for a simulation including generating random weights ''')
    parser.add_argument('-label', dest='label', type=str, help='Name of the root directory of the directory strucuture that is created', required = True)
    args = parser.parse_args(sys.argv[1:])
    label = args.label

    if os.path.exists('./simulations/%s' % label):
        raise Exception('Directory already exists! State a different label or delete direcotry!')
    
    print('Creating directory structure...')
    subfolder = ['plots', 'weights', 'activity', 'random', 'meta']
    for subf in subfolder:
        os.makedirs('./simulations/%s/%s' % (label, subf))

    create_weights(label)
    










