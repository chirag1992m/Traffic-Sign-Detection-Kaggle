#!/bin/bash

#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=10:00:00
#PBS -l mem=12GB
#PBS -N traffic_sign
#PBS -M chirag.m@nyu.edu
#PBS -j oe
#PBS -m abe

#Change the name of job (under -N) and options to the actual run

#Purge all the loaded modules, we'll load only the required modules
module purge

#Define important directories
SRCDIR=$WORK
PROJECT_NAME=Traffic-Sign-Detection-Kaggle
RUNDIR=$SCRATCH

#do the work
#Move to running directory
echo "Moving to running directory $RUNDIR"
cd $RUNDIR

#Load the required modules
echo "Loading the modules..."

echo "Loading torch/gnu/20160623"
module load torch/gnu/20160623

echo "Modules loaded!"

#Copy code to running directory
if [ ! -d "$PROJECT_NAME" ]; then
	echo "Copying code from $WORK/$PROJECT_NAME to $RUNDIR..."
	cp -r $WORK/$PROJECT_NAME $RUNDIR/
	echo "Copying done!"
else
	echo "Syncing code from $WORK/$PROJECT_NAME to $RUNDIR..."
	rsync -Cvrzh --exclude='*.ppm' /$WORK/$PROJECT_NAME $RUNDIR/
	echo "Syncing done!"
fi

#Move to project directory
echo "Moving to project directory.."
cd $PROJECT_NAME

echo "Running the model..."
qlua main.lua -model cifar -verbose -cuda -suffix 'baseline' #Change the options to run here
echo "All Done!"