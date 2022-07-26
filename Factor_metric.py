# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:19:55 2020

@author: gxjco
"""

import argparse
import numpy as np
from sklearn import linear_model

def prune_dims(variances, threshold=0.005):
  """Mask for dimensions collapsed to the prior."""
  scale_z = np.sqrt(variances)
  return scale_z >= threshold

def factor_metric_compute(type_, embeddings_files_list=None, labels_files_list=None, save_file=None):
  if embeddings_files_list is None and labels_files_list is None:
    path='/home/csolis/forked_repo_nedvae/quantitative_evaluation/'
    z_a1=np.load(path+type_+'_WS_graph_testing_a1_z.npy')
    z_a2=np.load(path+type_+'_WS_graph_testing_a2_z.npy')
    z_b1=np.load(path+type_+'_WS_graph_testing_b1_z.npy')
    z_b2=np.load(path+type_+'_WS_graph_testing_b2_z.npy')
    z_c1=np.load(path+type_+'_WS_graph_testing_c1_z.npy')
    z_c2=np.load(path+type_+'_WS_graph_testing_c2_z.npy')

  

    train_labels=np.concatenate((np.zeros(250),np.ones(250),np.ones(250)*2),axis=0)
    test_labels=np.concatenate((np.zeros(250),np.ones(250),np.ones(250)*2),axis=0)
    train_samples=np.concatenate((z_a1[:250],z_b1[:250],z_c1[:250]),axis=0)[:,:]
    test_samples=np.concatenate((z_a1[250:],z_b1[250:],z_c1[250:]),axis=0)[:,:]


  else:
    print('Appending embeddings from files ....')
    print(embeddings_files_list)
    embs_train = []
    embs_test = []
    labels_train = []
    labels_test = []
    for (embeddings_file, labels_file) in zip(embeddings_files_list, labels_files_list):
      emb = np.load(embeddings_file)
      
      lbl = np.load(labels_file)

      embs_train.append(emb[:int(lbl.shape[0]/2)])
      embs_test.append(emb[int(lbl.shape[0]/2):])
      labels_train.append(lbl[:int(lbl.shape[0]/2)])
      labels_test.append(lbl[int(lbl.shape[0]/2):])

    embs_train = np.array(embs_train) #(num of files (6), 500/2, batch_size (25), num_dims (9))
    labels_train = np.array(labels_train, dtype=np.float64)# num_files, 500/2 (total num of graphs per file (12500) div by batch size (25))
    embs_test = np.array(embs_test) 
    labels_test = np.array(labels_test, dtype=np.float64)

    train_labels = np.reshape(labels_train, (labels_train.shape[0]*labels_train.shape[1]))
    test_labels = np.reshape(labels_test, (labels_test.shape[0]*labels_test.shape[1]))
    train_samples = np.reshape(embs_train, (embs_train.shape[0]*embs_train.shape[1],embs_train.shape[2],embs_train.shape[3]))
    test_samples = np.reshape(embs_test, (embs_test.shape[0]*embs_test.shape[1],embs_test.shape[2],embs_test.shape[3]))

  D=z_a1.shape[2]
  L=int(max(train_labels))+1#3 #3 factors
  print(f'L is {L}')
  
  global_var=np.var(train_samples.reshape([-1,9]),axis=0)
  active_dim=prune_dims(global_var)

  #generate each batch
  for i in range(500):
    train_samples[i,:,:]/=np.std(train_samples[i,:,:])
    test_samples[i,:,:]/=np.std(test_samples[i,:,:])
    
  train_var=np.var(train_samples,axis=1)
  test_var=np.var(test_samples,axis=1)
  train_index=np.zeros(len(train_var))
  test_index=np.zeros(len(test_var))
  
  for i in range(len(train_var)):
    train_index[i]=np.argmin(train_var[i][active_dim])
    test_index[i]=np.argmin(test_var[i][active_dim])

  #votes
  training_votes=np.zeros((D,L))
  testing_votes=np.zeros((D,L))
  for i in range(len(train_index)):
      training_votes[int(train_index[i]),int(train_labels[i])]+=1
      testing_votes[int(test_index[i]),int(test_labels[i])]+=1
    
  #classifier    
  C=np.argmax(training_votes,axis=1)
  other_index = np.arange(training_votes.shape[0])

  #evaluate
  train_accuracy = np.sum(training_votes[other_index,C]) * 1. / np.sum(training_votes)
  test_accuracy = np.sum(testing_votes[other_index,C]) * 1. / np.sum(testing_votes)
  print(type_+"factor Training set accuracy: ", train_accuracy)
  print(type_+"factor Evaluation set accuracy: ", test_accuracy)
  if save_file is not None:
    file1 = open(save_file, "a")
    file1.write(str(save_file)+ '\n')
    file1.write('Factor-vae score:'+str(test_accuracy) + '\n')
  return test_accuracy


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--embeddings_files', nargs="+", help='list of embeddings_files')
  parser.add_argument('--labels_files', nargs="+", help='list of labels files')
  parser.add_argument('--save_file', type=str, help='path to file where to store score')

  args = parser.parse_args()
  print('Computinf factor vae score')
  factor_metric_compute(type_='FactorVAE', embeddings_files_list=args.embeddings_files, labels_files_list=args.labels_files, save_file=args.save_file)









