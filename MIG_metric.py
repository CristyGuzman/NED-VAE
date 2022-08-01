# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:21:13 2020

@author: gxjco
"""
import numpy as np
import argparse
import sklearn.metrics
from sklearn.preprocessing import KBinsDiscretizer

def discrete_mutual_info(mus, ys,num_bin):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
          m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :],  make_discretizer(mus[i, :],num_bin))   
  return m

def discrete_entropy(ys,num_bin):
  """Compute discrete entropy of the factors."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(make_discretizer(ys[j, :],num_bin),make_discretizer(ys[j, :],num_bin))
  return h

def make_discretizer(target, num_bins):
    """Wrapper that creates discretizers."""
    Dis=KBinsDiscretizer(num_bins, encode='ordinal').fit(target.reshape(-1,1))
    
    return Dis.transform(target.reshape(-1,1)).reshape(-1)

def MIG_compute(type_, embeddings_file=None, factors_file=None, save_file=None):
  num_bin=10
  if embeddings_file is None and factors_file is None:
    path='/home/csolis/forked_repo_nedvae/quantitative_evaluation/'
    factor=np.transpose(np.load(path+'WS_factor_testing2.npy'))
    code=np.transpose(np.load(path+type_+'_WS_graph_testing2_z.npy').reshape(-1,9))
  else:
    else:
    factor=np.transpose(np.load(embeddings_file).reshape(-1,9))
    code=np.transpose(np.load(factors_file).reshape(-1,9))

  m = discrete_mutual_info(code, factor,num_bin)
  # m is [num_latents, num_factors]
  entropy_h = discrete_entropy(factor,num_bin)
  sorted_m = np.sort(m, axis=0)[::-1]
  print('MIG score: '+str(np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy_h[:]))))
  return np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy_h[:]))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--embeddings_file', type=str, help='list of embeddings_files')
  parser.add_argument('--factors_file', type=str, help='list of labels files')
  parser.add_argument('--save_file', type=str, help='path to file where to store score')

  args = parser.parse_args()

  MIG_compute(type_='FactorVAE ', embeddings_file=args.embeddings_file, factors_file=args.factors_file, save_file=args.save_file)

