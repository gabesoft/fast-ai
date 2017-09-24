
python: export KERAS_BACKEND := theano
python: export PYTHONSTARTUP := $(CURDIR)/.pythonrc

# Start the python interpreter with the correct environment
python:
	python

# Activate the conda environment
activate:
	source activate myenv

# Start ipython
ipython: export KERAS_BACKEND := theano
ipython:
	ipython --TerminalInteractiveShell.editing_mode=vi --matplotlib

# Start a nix shell with miniconda
conda-shell:
	nix-shell ./.conda-shell.nix