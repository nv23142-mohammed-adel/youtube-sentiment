##Environment setup


conda create -n youtube python=3.11 -c main -y

conda activate youtube


##Dependency Installation


pip install -r requirements.txt


##DVC


dvc init

dvc repro

dvc dag


##Cloud Configuration(AWS)


aws configure
