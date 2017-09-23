
python: export KERAS_BACKEND := theano
python: export PYTHONSTARTUP := $(CURDIR)/.pythonrc

# Start the python interpreter with the correct environment
python:
	python