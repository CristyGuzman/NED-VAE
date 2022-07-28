#!/bin/bash

set -e
set -o xtrace
echo "first variable is metric $1"
echo "second variable is dimension $2"

for MODEL_I in FactorVAE_1658867433_10 FactorVAE_1658878251_20 FactorVAE_1658878306_1 FactorVAE_1658878329_10 FactorVAE_1658878488_20 FactorVAE_1658878564_40 FactorVAE_1658879092_1 FactorVAE_1658879962_10 FactorVAE_1658880548_20 FactorVAE_1658880677_40 FactorVAE_1658880769_1 FactorVAE_1658880919_10 FactorVAE_1658880930_20 FactorVAE_1658881391_40
do
    for MODEL_J in FactorVAE_1658867433_10 FactorVAE_1658878251_20 FactorVAE_1658878306_1 FactorVAE_1658878329_10 FactorVAE_1658878488_20 FactorVAE_1658878564_40 FactorVAE_1658879092_1 FactorVAE_1658879962_10 FactorVAE_1658880548_20 FactorVAE_1658880677_40 FactorVAE_1658880769_1 FactorVAE_1658880919_10 FactorVAE_1658880930_20 FactorVAE_1658881391_40
    do
        sbatch -p gpu --cpus-per-task 2 --mem-per-cpu 8G --gres=gpu:1 --wrap "/home/csolis/forked_repo_nedvae/NED-VAE/model_centrality.sh $MODEL_I $MODEL_J $1 $2"
   
    done
done