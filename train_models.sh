#!/bin/bash

for seed in {1..3}
do
  echo $seed
  for beta in 1 10 20 40
  do
     sbatch -p gpu --cpus-per-task 2 --mem-per-cpu 5G --gres=gpu:1 --wrap "python /home/csolis/forked_repo_nedvae/NED-VAE/main.py --vae_type FactorVAE --type train --beta=$beta"
     
     
  done
done