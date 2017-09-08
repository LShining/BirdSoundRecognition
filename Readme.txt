Index

BirdATraining
	A folder that contains a sample set of data. Can be feature-extracted by "featureExtraction.py"

project_data
	A folder that contains all extracted features.

featureExtraction.py:
	Feature extraction script as its name describes.
	take samples01.wav,samples02.wav,etc in the same directory
	output samples01.npy,...,etc
	output BirdATraining.npy as obsevation sequence for training
	output multiple test.npy as test sample
	#must be manually specified for number of samples and tests and bird Code(A,B,C,etc)

HMM Training And Evaluation.ipynb:
	IPython Notebook as its name shows. 
	To open this file:
		1.Open a terminal/command line in the directory
		2.use command "jupyter notebook" (without quotes) to summon a browser client
		3.Open the notebook in browser summoned.

peakdetect.py
	Peakdetect function translated from the same Matlab function

plots.ipynb
	Notebook file used to generate the figures for extraction methods.

thinkdsp.py
	Package used for generate a spectrogram from .wav files
	Adapted by us for this specific case.

thinkplot.py
	Package comes together with thinkdsp.py

thinkstats2.py
	Package comes together with thinkdsp.py


Guides for packages:
	most packages(and pip/jupyter) are included in Anaconda except hmmlearn. Use "pip install hmmlearn"(without quotes) to install the hmmlearn package on the computer.