# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:57:13 2020

@author: gxjco
"""

from __future__ import division
from __future__ import print_function

import time
import os
import sys

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from sklearn import manifold
from scipy.special import expit

from optimizer import OptimizerVAE
from model import *
from preprocessing import *

from beta_metric import*
from DCI_metric import*
from Factor_metric import*
from MIG_metric import*
from Mudularity import*
from SAP_metric import*

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Settings
for name in list(flags.FLAGS):
      delattr(flags.FLAGS,name)
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden11', 10, 'Number of units in hidden layer 11.')
flags.DEFINE_integer('hidden12', 8, 'Number of units in hidden layer 12.')
flags.DEFINE_integer('hidden13', 3, 'Number of units in hidden layer 13.')
flags.DEFINE_integer('hidden21', 10, 'Number of units in hidden layer 21.')
flags.DEFINE_integer('hidden22', 8, 'Number of units in hidden layer 22.')
flags.DEFINE_integer('hidden23', 6, 'Number of units in hidden layer 23.')
flags.DEFINE_integer('hidden24', 3, 'Number of units in linear layer 24.')
flags.DEFINE_integer('hidden31', 10, 'Number of units in hidden layer 31.')
flags.DEFINE_integer('hidden32', 8, 'Number of units in linear layer 32.')
flags.DEFINE_integer('hidden33', 6, 'Number of units in hidden layer 31.')
flags.DEFINE_integer('hidden34', 3, 'Number of units in linear layer 32.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('edge_dropout', 0., 'Dropout for individual edges in training graph')
flags.DEFINE_float('autoregressive_scalar', 0., 'Scalar for Graphite')
flags.DEFINE_integer('vae', 1, 'for variational objective')
flags.DEFINE_integer('batch_size', 25, 'Number of samples in a batch.')
flags.DEFINE_integer('decoder_batch_size',25, 'Number of samples in a batch.')
flags.DEFINE_integer('subsample', 0, 'Subsample in optimizer')
flags.DEFINE_float('subsample_frac', 1, 'Ratio of sampled non-edges to edges if using subsampling')
flags.DEFINE_integer('num_feature', 1, 'Number of features.')
flags.DEFINE_integer('verbose', 1, 'Output all epoch data')
flags.DEFINE_integer('test_count', 10, 'batch of tests')
flags.DEFINE_string('model', 'feedback', 'Model string.')
flags.DEFINE_integer('seeded', 1, 'Set numpy random seed')
flags.DEFINE_integer('connected_split', 1, 'use split with training set always connected')
flags.DEFINE_string('type', 'test', 'train or test')
flags.DEFINE_integer('if_visualize', 0, 'varying the z to see the generated graphs')
flags.DEFINE_string('model_dir', '/home/csolis/forked_repo_nedvae/models', 'model to be loaded for evaluation')
flags.DEFINE_integer('generate_graphs', 0, 'if 1, generates graphs from specified folder in model_file arg')
flags.DEFINE_integer('fixed_factor', 7, 'fixes node/graph/edge embedding to same value')
flags.DEFINE_string('model_file', '/home/csolis/forked_repo_nedvae/NED-VAE/models/beta-VAE/1657141272model_dgt_global_950.ckpt', 'model directory')
flags.DEFINE_integer('model_centrality', 0, 'Indicates if embeddings should be stored for computing model centrality')
flags.DEFINE_integer('fix_factor_values', 0, 'If 1, fixes values in dimension givne by FLAGS.fixed_factor')
flags.DEFINE_integer('num_factors', 3, 'For now only accepts either 9 or 3 factors, number of generative factors.')

def ZscoreNormalization(x, mean_, std_):
    """Z-score normaliaztion"""
    x = (x - mean_) / std_
    return x

def load_data_syn_test(dataset_file1,dataset_file2):
    adj=np.load(dataset_file1)
    node=np.load(dataset_file2)
    return np.array(adj),node.reshape(adj.shape[0],adj.shape[1],-1,1)

def main(beta,dataset_file1,dataset_file2,type_model):
        if 'vae_type' in list(flags.FLAGS):
            delattr(flags.FLAGS,'vae_type')
        flags.DEFINE_string('vae_type', type_model, 'local or global or local_global')
        if FLAGS.seeded:
            np.random.seed(1)
        
          # Load data
        adj,node= load_data_syn_test(dataset_file1,dataset_file2)
        adj_orig = adj
        adj_train=adj

        #if FLAGS.features == 0:
         #     feature_test = np.tile(np.identity(adj_test.shape[1]),[adj_test.shape[0],1,1])
          #    feature_train = np.tile(np.identity(adj_train.shape[1]),[adj_train.shape[0],1,1])
              
        #feature_train=features[:]
        #feature_test=features[:]      
            # featureless
        num_nodes = adj.shape[1]
        
        #features = sparse_to_tuple(features.tocoo())
        num_features = node.shape[2]
        pos_weight = float(adj.shape[0] *adj.shape[1] * adj.shape[1] - adj.sum()) / adj.sum()
        norm = adj.shape[0] *adj.shape[1] * adj.shape[1] / float((adj.shape[0] *adj.shape[1] * adj.shape[1] - adj.sum()) * 2)
        
        adj_orig=adj_train.copy()
        for i in range(adj_train.shape[0]):
            adj_orig[i] = adj_train[i].copy() + np.eye(adj_train.shape[1])
            
        #use encoded label
        adj_label=np.zeros(([adj_train.shape[0],adj_train.shape[1],adj_train.shape[2],2]))  
        for i in range(adj_train.shape[0]):
                for j in range(adj_train.shape[1]):
                    for k in range(adj_train.shape[2]):
                        adj_label[i][j][k][int(adj_orig[i][j][k])]=1
        
        placeholders = {
                'features': tf.compat.v1.placeholder(tf.float32,[FLAGS.batch_size,node.shape[1],node.shape[2],node.shape[3]]),
                'adj': tf.compat.v1.placeholder(tf.float32,[FLAGS.batch_size,adj_train.shape[1],adj_train.shape[2]]),
                'adj_orig': tf.compat.v1.placeholder(tf.float32,[FLAGS.batch_size,adj_train.shape[1],adj_train.shape[2],2]),
                'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
                'i': tf.compat.v1.placeholder_with_default(0, shape=()),
            }
        
        #model = GCNModelFeedback(placeholders, num_features, num_nodes)
        model = GCNModelVAE(placeholders, num_features, num_nodes)
        
       
        if FLAGS.type=='train':
          with tf.name_scope('optimizer'):
                opt = OptimizerVAE(preds_edge=model.rec_edge_logits,
                                   preds_node=model.rec_node,
                                   labels_edge=tf.reshape(placeholders['adj_orig'], [-1,2]),
                                   labels_node=tf.reshape(placeholders['features'], [-1,1]),
                                   model=model, num_nodes=num_nodes,
                                   pos_weight=pos_weight,
                                   norm=norm,
                                   beta=beta)
        
     
        saver = tf.compat.v1.train.Saver()
            
        def generate_new(adj_test,adj_label,features,i):
           feed_dict = construct_feed_dict(adj_test, adj_label, features, placeholders)
           feed_dict.update({placeholders['dropout']: 0})
           feed_dict.update({placeholders['i']: i})
           if FLAGS.fix_factor_values == 1:
               i_model, z_n_orig, z_n_fixed, z_e_orig, z_e_fixed, z_g_orig, z_g_fixed, z_n,z_e,z_g,g,node = sess.run([model.i, model.z_n_orig, model.z_n2_fixed, model.z_e_orig, model.z_e2_fixed, model.z_g_orig, model.z_g2_fixed, model.z_mean_n,model.z_mean_e,model.z_mean_g, model.sample_rec_edge,model.sample_rec_node], feed_dict=feed_dict)
               return i_model, z_n_orig, z_n_fixed, z_e_orig, z_e_fixed, z_g_orig, z_g_fixed, z_n,z_e,z_g,g,node  
           else:
               z_n,z_e,z_g,g,node = sess.run([model.z_mean_n,model.z_mean_e,model.z_mean_g, model.sample_rec_edge,model.sample_rec_node], feed_dict=feed_dict)
               return z_n,z_e,z_g,g,node   
            
        if FLAGS.type=='test':
          with tf.compat.v1.Session() as sess:
            saver.restore(sess, FLAGS.model_file)
            print("Model restored.")
            graphs=[]
            nodes=[]
            z_n=[]
            z_e=[]
            z_g=[]
            labels=[]
            test_batch_num=int(adj_train.shape[0]/FLAGS.batch_size)
            for i in range(test_batch_num):
                labels.append(i%FLAGS.num_factors)
                adj_batch_test=adj_train[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                adj_batch_label=adj_label[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                feature_batch_test=node[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size]
                val = int(i%FLAGS.num_factors)
                if FLAGS.fix_factor_values == 1:
                    i_model, z_n_orig, z_n_fixed, z_e_orig, z_e_fixed, z_g_orig, z_g_fixed, z_n_batch,z_e_batch,z_g_batch,g,n=generate_new(adj_batch_test,adj_batch_label,feature_batch_test,val)
                    #print(f'model.i is {i_model}')
                else:
                    z_n_batch,z_e_batch,z_g_batch,g,n=generate_new(adj_batch_test,adj_batch_label,feature_batch_test,val)
                graphs.append(g)
                nodes.append(n)
                z_n.append(z_n_batch)
                z_e.append(z_e_batch)
                z_g.append(z_g_batch)
            graphs=np.array(graphs).reshape(-1,num_nodes,num_nodes)
            nodes=np.array(nodes).reshape(-1,num_nodes)
            z_n=np.array(z_n)
            z_e=np.array(z_e)
            z_g=np.array(z_g)
            z=np.concatenate((z_n,z_e,z_g),axis=2)
            l=np.array(labels)
            type_name=dataset_file1.split('/')[-1].split('.')[0]
            np.save('./quantitative_evaluation/'+FLAGS.vae_type+'_'+type_name+'_z.npy',z)
            if (FLAGS.if_visualize==0 and FLAGS.generate_graphs==1): 
                if FLAGS.fix_factor_values == 1:
                    file_name = 'fixed_'+str(FLAGS.num_factors)
                else:
                    file_name = '_'
                try:
                    graphs_path = '/home/csolis/forked_repo_nedvae/generated_graphs/' + type_name + '_' + file_name + '_' + 'generated_graphs.npy'
                    nodes_path = '/home/csolis/forked_repo_nedvae/generated_nodes/' + type_name + '_' + file_name + '_' + 'generated_nodes.npy'
                    os.remove(graphs_path)
                    os.remove(nodes_path)
                except OSError:
                    pass
                np.save('/home/csolis/forked_repo_nedvae/generated_graphs/' + type_name + '_' + file_name + '_' + "generated_graphs.npy", graphs)
                print(f'Saved graphs to /home/csolis/forked_repo_nedvae/generated_graphs/{type_name}_{file_name}_generated_graphs.npy')
                np.save('/home/csolis/forked_repo_nedvae/generated_nodes/' + type_name + '_' + file_name + '_' + "generated_nodes.npy", nodes)
                print(f'Saved nodes to /home/csolis/forked_repo_nedvae/generated_graphs/{type_name}_{file_name}_generated_nodes.npy')
            elif FLAGS.model_centrality == 1:
                model_name = dataset_file1.split('/')[-1].split('.')[0]
                name = FLAGS.model_file.split('/')[-1].split('.')[0]
                print(f'Saving embeddings generated from {name} using graphs generated from {model_name}.')
                try:
                    z_path = '/home/csolis/forked_repo_nedvae/embeddings/' + name  + '_z.npy'
                    labels_path='/home/csolis/forked_repo_nedvae/labels/' + name + '_labels.npy'
                    os.remove(z_path)
                    os.remove(labels_path)
                except OSError:
                    pass
                print(f'Saving embeddings in /home/csolis/forked_repo_nedvae/embeddings/{name}_z.npy')
                print(f'Saving labels in /home/csolis/forked_repo_nedvae/labels/{name}_labels.npy')
                np.save('/home/csolis/forked_repo_nedvae/embeddings/' + name  + '_z.npy',z)
                np.save('/home/csolis/forked_repo_nedvae/labels/' + name + '_labels.npy', l)
            else:
                print('Not generating graphs, not com[uting cross evaluations for model centrality. Done.')
                    

def mc_matrices(*model_names):    
    models = [str(m) for m in models_names]
    models_dict = {j:i for i,j in enumerate(models)}

    factor_mc_matrix = np.zeros((len(models_dict), len(models_dict)))
    d_mc_matrix = np.zeros((len(models_dict), len(models_dict)))
    c_mc_matrix = np.zeros((len(models_dict), len(models_dict)))
    i_mc_matrix = np.zeros((len(models_dict), len(models_dict)))
    for model_i in models:
        for model_j in models:
            pth_factor = '/home/csolis/forked_repo_nedvae/mc_scores/FactorVAE_'+str(model_i)+'_FactorVAE_'+str(model_j)+'_FactorVAE_3.txt'
            pth_dci = '/home/csolis/forked_repo_nedvae/mc_scores/FactorVAE_'+str(model_i)+'_FactorVAE_'+str(model_j)+'_DCI_9.txt'
            with open(pth_factor) as file:
                lines = file.readlines()
                lines_f = [line.rstrip() for line in lines] 
            with open(pth_dci) as file:
                lines = file.readlines()
                lines_dci = [line.rstrip() for line in lines] 
            factor_mc_matrix[models_dict[model_i], models_dict[model_j]] = float(lines_f[-1].split(':')[-1])
            d_mc_matrix[models_dict[model_i], models_dict[model_j]] = float(lines_dci[1].split(':')[-1])
            c_mc_matrix[models_dict[model_i], models_dict[model_j]] = float(lines_dci[2].split(':')[-1])
            i_mc_matrix[models_dict[model_i], models_dict[model_j]] = float(lines_dci[3].split(':')[-1])
    np.save('/home/csolis/forked_repo_nedvae/mc_matrices/factor_mc.npy',factor_mc_matrix)
    np.save('/home/csolis/forked_repo_nedvae/mc_matrices/disentanglement_mc.npy',d_mc_matrix)
    np.save('/home/csolis/forked_repo_nedvae/mc_matrices/completeness_mc.npy',c_mc_matrix)
    np.save('/home/csolis/forked_repo_nedvae/mc_matrices/informativeness_mc.npy',i_mc_matrix)         

if __name__ == '__main__':
    #types=['beta-VAE','DIP-VAE','InfoVAE','FactorVAE','HFVAE']
    types=['FactorVAE']
    for t in types:
         tf.compat.v1.reset_default_graph()
         main(20,'./graph_generator/WS_graph_testing_a1.npy','./graph_generator/WS_graph_testing_a1_nodes.npy',t)
         tf.compat.v1.reset_default_graph()
         main(20,'./graph_generator/WS_graph_testing_a2.npy','./graph_generator/WS_graph_testing_a2_nodes.npy',t)
         tf.compat.v1.reset_default_graph()
         main(20,'./graph_generator/WS_graph_testing_b1.npy','./graph_generator/WS_graph_testing_b1_nodes.npy',t)
         tf.compat.v1.reset_default_graph()
         main(20,'./graph_generator/WS_graph_testing_b2.npy','./graph_generator/WS_graph_testing_b2_nodes.npy',t)
         tf.compat.v1.reset_default_graph()
         main(20,'./graph_generator/WS_graph_testing_c1.npy','./graph_generator/WS_graph_testing_c1_nodes.npy',t)
         tf.compat.v1.reset_default_graph()
         main(20,'./graph_generator/WS_graph_testing_c2.npy','./graph_generator/WS_graph_testing_c2_nodes.npy',t)
         tf.compat.v1.reset_default_graph()
         main(20,'./graph_generator/WS_graph_testing2.npy','./graph_generator/WS_node_testing2.npy',t)
         
         if FLAGS.generate_graphs==0:
             beta_score=beta_metric_compute(t)
             f_score=factor_metric_compute(t)
             MIG_score=MIG_compute(t)
             infor_score,disen_score,comp_score=DCI_metric_compute(t)
             mod_score,exp_score=modularity_compute(t)
             sap_score=SAP_compute(t)
            
             file1 = open('./result_sythetic2.txt', "a")
             file1.write(t+': beta_score'+str(beta_score) + '\n')
             file1.write(t+': f_score'+str(f_score) + '\n')
             file1.write(t+': MIG_score'+str(MIG_score) + '\n')
             file1.write(t+': infor_score'+str(infor_score) + '\n')
             file1.write(t+': disen_score'+str(disen_score) + '\n')
             file1.write(t+': comp_score'+str(comp_score) + '\n')
             file1.write(t+': mod_score'+str(mod_score) + '\n')
             file1.write(t+': exp_score'+str(exp_score) + '\n')
             file1.write(t+': sap_score'+str(sap_score) + '\n')
             file1.close()
         