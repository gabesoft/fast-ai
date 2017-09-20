
python: export KERAS_BACKEND := theano
python: export PYTHONSTARTUP := $(CURDIR)/.pythonrc
# python: export _IMAGE_DATA_FORMAT := 'channels_first'

# Start the python interpreter with the correct environment set up
python:
	python