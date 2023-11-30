#!/bin/bash

export TMPDIR=$SCRATCHDIR

export PYTHONUSERBASE="/storage/projects/msml/NeuralSpecLib/python310"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python3.10/site-packages:${PYTHONPATH}"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"

python3 /storage/projects/msml/NeuralSpecLib/NeuralSpecLibrary/main.py