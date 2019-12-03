conda create -y --name MLMI python=3.7
conda activate MLMI
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install numpy
conda install matplotlib
conda install tensorboard
conda install -c conda-forge jupyterlab
