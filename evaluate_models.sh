#!/bin/bash


## For generating embeddings from original test graphs, run evaluate.py with 
# model_centrality=0
# fix_factor_values=0 
# generate_graphs=0 
# model_file=PATH/TO/MODEL.ckpt

set -e
set -o xtrace
while [ $# -gt 0 ] 
do
    echo $1
    sbatch -p gpu --cpus-per-task 2 --mem-per-cpu 8G --gres=gpu:1 --wrap "python /home/csolis/forked_repo_nedvae/NED-VAE/evaluate.py --vae_type FactorVAE --model_centrality 0 --fix_factor_values 0 --generate_graphs 0 --model_file $1"
    shift
done
