# Environment setup


conda create -n youtube python=3.11 -c main -y


## Initialize the shell


conda init bash


## Activate the environment


conda activate youtube


## Dependency Installation


pip install -r requirements.txt


# DVC


## Initialize DVC


dvc init -f


## run pipeline


dvc repro


dvc dag


# Cloud Configuration(AWS)


aws configure
