#!/bin/sh 


### General options

### –- specify queue --
#BSUB -q gpuv100
##SUB -q gpua100

### -- set the job Name --
#BSUB -J Run_0-9b

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# request system-memory (per core)
#BSUB -R "rusage[mem=4GB]"

### -- Specify how the cores are distributed across nodes --
# The following means that all the cores must be on one single host
#BSUB -R "span[hosts=1]"

### -- Specify the output and error file. %J is the job-id --
#BSUB -o out/jobs/iris_%J.out
#BSUB -e out/jobs/iris_%J.err

# -- end of LSF options --
NAME=Run_0-9b
LR=0.0002
EPOCHS=8
BS=16
DATASET=both_eyes_together

MODEL=VIT-21k
REPO=${HOME}/biometrics

OUT=${REPO}/black-hole/out/${MODEL}/${DATASET}/${NAME}
mkdir -p ${OUT}

# Activate venv
#module load python3/3.10.14
source ${REPO}/biometrics/bin/activate


##### TRAINING #####
python3 ${REPO}/src/train_model.py \
    --out-path ${OUT} \
    --epochs ${EPOCHS} \
    --learning-rate ${LR} \
    --batch-size ${BS} \
    --data-set ${DATASET}    > ${OUT}/log.out

##### EVALUATION #####
python3 ${REPO}/src/evaluate_model.py  \
    --out-path ${OUT} \
    --data-set ${DATASET} --run-inference   >> ${OUT}/log.out
 
