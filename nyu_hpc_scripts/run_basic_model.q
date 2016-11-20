#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=05:00:00
#PBS -l mem=2GB
#PBS -N basic_model
#PBS -M chirag.m@nyu.edu
#PBS -j oe
#PBS -o logs/${PBS_JOBID.log}

#Purge all the loaded modules, we'll load only the required modules
module purge

#Load required modules
module load torch/gnu/20160623

SRCDIR=$WORK/Traffic-Sign-Detection-Kaggle

#Start running the job
cd $SRCDIR
qlua main.lua -model cifar