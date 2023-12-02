#!/bin/bash
#PBS -N HOMO-LUMO_regression
#PBS -q gpu
#PBS -l select=1:ncpus=2:ngpus=1:mem=34gb:scratch_local=128gb:gpu_cap=cuda61
#PBS -l walltime=3:00:00
#PBS -m ae

export TMPDIR=$SCRATCHDIR

cd $SCRATCHDIR

wget https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz

tar -xzf Python-3.10.2.tgz

cd Python-3.10.2
./configure --prefix=$SCRATCHDIR/python310
make
make install

# Set up the Python environment variables
export PYTHONUSERBASE="$SCRATCHDIR/python310"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.10/site-packages:$PYTHONPATH"
export PATH="$PYTHONUSERBASE/bin:$PATH"

"$PYTHONUSERBASE/bin/pip3" install -r /storage/projects/msml/NeuralSpecLib/NeuralSpecLibrary/requirements.txt

singularity run --nv -B $SCRATCHDIR /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:23.05-py3.SIF python3 /storage/projects/msml/NeuralSpecLib/NeuralSpecLibrary/main.py
