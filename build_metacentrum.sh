

# shellcheck disable=SC2164
cd /storage/projects/msml/NeuralSpecLib/
wget https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz

tar -xzf Python-3.10.2.tgz


./configure --prefix=/storage/projects/msml/NeuralSpecLib/python310
make
make install