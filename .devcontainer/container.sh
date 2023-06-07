python -m pip install --upgrade pip
python -m pip install 'flit>=3.8.0'
python -m flit install --only-deps --deps develop
echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc
