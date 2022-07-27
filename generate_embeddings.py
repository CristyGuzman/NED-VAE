"""
Generate embeddings from graph and node data generated from decoder model. Must return embeddings with their corresponding labels.

"""

import os
import tensorflow as tf
import numpy as np
from input_data import load_data_syn
from evaluate import main


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('graphs_file', '/home/csolis/forked_repo_nedvae/NED-VAE/models/beta-VAE/1657141272model_dgt_global_950_WS_graph_testing_c2_generated_graphs.npy', 'File with generated graphs.')
flags.DEFINE_string('nodes_file', '/home/csolis/forked_repo_nedvae/NED-VAE/models/beta-VAE/1657141272model_dgt_global_950_WS_graph_testing_c2_generated_nodes.npy', 'File with generated nodes.')


if __name__ == '__main__':
    types = ['FactorVAE']
    for t in types:
        tf.compat.v1.reset_default_graph()
        main(20, FLAGS.graphs_file, FLAGS.nodes_file, t)