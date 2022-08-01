#!/bin/bash

set -o xtrace

MODEL_I=$1
MODEL_J=$2
METRIC=$3
DIMS=$4

MODEL_I_DIR="/home/csolis/forked_repo_nedvae/models/${MODEL_I}.ckpt"
MODEL_J_DIR="/home/csolis/forked_repo_nedvae//models/${MODEL_J}.ckpt"
SAVE_MC_DIR="/home/csolis/forked_repo_nedvae/mc_scores/${MODEL_I}_${MODEL_J}_${METRIC}_${DIMS}.txt"

[ $METRIC == "FactorVAE" ] && FIX_FACTOR=1 || FIX_FACTOR=0
echo "fixing factor set to $FIX_FACTOR"
echo "model j stored in  $MODEL_J_DIR"

python /home/csolis/forked_repo_nedvae/NED-VAE/evaluate.py --vae_type FactorVAE --model_centrality 0 --fix_factor_values $FIX_FACTOR --generate_graphs 1 --model_file $MODEL_J_DIR --num_factors $DIMS


if [ $METRIC == "FactorVAE" ] ; then
GRAPHS_FILE="/home/csolis/forked_repo_nedvae/generated_graphs/WS_graph_testing2_fixed_${DIMS}_generated_graphs.npy" 
NODES_FILE="/home/csolis/forked_repo_nedvae/generated_nodes/WS_graph_testing2_fixed_${DIMS}_generated_nodes.npy" 

python /home/csolis/forked_repo_nedvae/NED-VAE/generate_embeddings.py --model_centrality 1 --fix_factor_values 0 --generate_graphs 0 --graphs_file $GRAPHS_FILE --nodes_file $NODES_FILE --model_file $MODEL_I_DIR --num_factors $DIMS


EMBEDDINGS="/home/csolis/forked_repo_nedvae/embeddings/${MODEL_I}_z.npy" 
LABELS="/home/csolis/forked_repo_nedvae/labels/${MODEL_I}_labels.npy" 

python /home/csolis/forked_repo_nedvae/NED-VAE/Factor_metric.py --embeddings_files $EMBEDDINGS --labels_files $LABELS --save_file $SAVE_MC_DIR

fi

if [ $METRIC == "DCI" ] ; then
GRAPHS_FILE="/home/csolis/forked_repo_nedvae/generated_graphs/WS_graph_testing2___generated_graphs.npy" 
NODES_FILE="/home/csolis/forked_repo_nedvae/generated_nodes/WS_graph_testing2___generated_nodes.npy" 

python /home/csolis/forked_repo_nedvae/NED-VAE/generate_embeddings.py --model_centrality 1 --fix_factor_values 0 --generate_graphs 0 --graphs_file $GRAPHS_FILE --nodes_file $NODES_FILE --model_file $MODEL_I_DIR --num_factors $DIMS


EMBEDDINGS="/home/csolis/forked_repo_nedvae/embeddings/${MODEL_I}_z.npy" 
FACTOR_EMBEDDINGS="/home/csolis/forked_repo_nedvae/quantitative_evaluation/FactorVAE_WS_graph_testing2_z.npy"

python /home/csolis/forked_repo_nedvae/NED-VAE/DCI_metric.py --embeddings_file $EMBEDDINGS --factors_file $FACTOR_EMBEDDINGS --save_file $SAVE_MC_DIR

fi

if [ $METRIC == "MIG" ] ; then
GRAPHS_FILE="/home/csolis/forked_repo_nedvae/generated_graphs/WS_graph_testing2___generated_graphs.npy" 
NODES_FILE="/home/csolis/forked_repo_nedvae/generated_nodes/WS_graph_testing2___generated_nodes.npy" 

python /home/csolis/forked_repo_nedvae/NED-VAE/generate_embeddings.py --model_centrality 1 --fix_factor_values 0 --generate_graphs 0 --graphs_file $GRAPHS_FILE --nodes_file $NODES_FILE --model_file $MODEL_I_DIR --num_factors $DIMS


EMBEDDINGS="/home/csolis/forked_repo_nedvae/embeddings/${MODEL_I}_z.npy" 
FACTOR_EMBEDDINGS="/home/csolis/forked_repo_nedvae/quantitative_evaluation/FactorVAE_WS_graph_testing2_z.npy"

python /home/csolis/forked_repo_nedvae/NED-VAE/MIG_metric.py --embeddings_file $EMBEDDINGS --factors_file $FACTOR_EMBEDDINGS --save_file $SAVE_MC_DIR

fi