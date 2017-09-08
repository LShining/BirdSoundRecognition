import thinkdsp as thinkDSP
import thinkplot
import matplotlib
import numpy as np
from peakdetect import *
# Feature extraction method 1
def preprocessA(sample):
    signal = thinkDSP.read_wave("samples"+sample+".wav")
    spectrum = signal.make_spectrum()
    spectrum.high_pass(cutoff=100, factor=0.1)
    signal = spectrum.make_wave()
    #signal.write("outcome.wav")
    spectrogram = signal.make_spectrogram(seg_length=4096)
    specMatrix = spectrogram.plot(returnArray=1,length=None,high=8000, cmap="gray")
    thinkplot.show()
    maxi = np.amax(specMatrix,axis = 0)
    indi = np.argmax(specMatrix,axis = 0)
    final = np.column_stack([maxi,indi])
    np.save("sample"+sample,final)

# Feature extraction method 2
def preprocessB(sample):
    signal = thinkDSP.read_wave("samples"+sample+".wav")
#    spectrum = signal.make_spectrum()
#    spectrum.high_pass(cutoff=100, factor=0.1)
#    signal = spectrum.make_wave()
    #signal.write("outcome.wav")
    spectrogram = signal.make_spectrogram(seg_length=4096)
    specMatrix = spectrogram.plot(returnArray=1,length=None,high=8000, cmap="gray")
#    thinkplot.show()
    
    maxi = np.amax(specMatrix,axis = 0)
    indi = np.argmax(specMatrix,axis = 0)
    
    maxValue = np.amax(specMatrix)
    specMatrix *= 200/maxValue
    output = []
    for timeIndex in range(specMatrix.shape[1]):
        #for timeIndex in range(2):
        quantile80IndexU = 0
        quantile60IndexU = 0
        quantile40IndexU = 0
        quantile20IndexU = 0
        quantile80IndexL = 0
        quantile60IndexL = 0
        quantile40IndexL = 0
        quantile20IndexL = 0
        column = specMatrix[:,timeIndex]
        for freqIndex in range(0,indi[timeIndex]):
            if (column[freqIndex]-0.8*maxi[timeIndex])*(column[freqIndex+1]-0.8*maxi[timeIndex])<0:
                quantile80IndexU = freqIndex
            if (column[freqIndex]-0.6*maxi[timeIndex])*(column[freqIndex+1]-0.6*maxi[timeIndex])<0:
                quantile60IndexU = freqIndex
            if (column[freqIndex]-0.4*maxi[timeIndex])*(column[freqIndex+1]-0.4*maxi[timeIndex])<0:
                quantile40IndexU = freqIndex
            if (column[freqIndex]-0.2*maxi[timeIndex])*(column[freqIndex+1]-0.2*maxi[timeIndex])<0:
                quantile20IndexU = freqIndex
        #'U' for 'Uppertail'.
        for freqIndex in range(indi[timeIndex],column.shape[0]-1):
            if (column[freqIndex]-0.8*maxi[timeIndex])*(column[freqIndex+1]-0.8*maxi[timeIndex])<0:
                quantile80IndexL = freqIndex+1
            if (column[freqIndex]-0.6*maxi[timeIndex])*(column[freqIndex+1]-0.6*maxi[timeIndex])<0:
                quantile60IndexL = freqIndex+1
            if (column[freqIndex]-0.4*maxi[timeIndex])*(column[freqIndex+1]-0.4*maxi[timeIndex])<0:
                quantile40IndexL = freqIndex+1
            if (column[freqIndex]-0.2*maxi[timeIndex])*(column[freqIndex+1]-0.2*maxi[timeIndex])<0:
                quantile20IndexL = freqIndex+1
        #'L' for 'Lowertail'.
        output.append([maxi[timeIndex],indi[timeIndex],quantile80IndexU,quantile60IndexU,quantile40IndexU,quantile20IndexU,quantile80IndexL,quantile60IndexL,quantile40IndexL,quantile20IndexL])
    np.save("samples"+sample,output)

# Feature extraction method 3
def preprocessC(sample):
    signal = thinkDSP.read_wave("samples"+sample+".wav")
#    spectrum = signal.make_spectrum()
#    spectrum.high_pass(cutoff=100, factor=0.1)
#    signal = spectrum.make_wave()
    #signal.write("outcome.wav")
    spectrogram = signal.make_spectrogram(seg_length=4096)
    specMatrix = spectrogram.plot(returnArray=1,length=None,high=8000, cmap="gray")
#    thinkplot.show()
    
#    maxi = np.amax(specMatrix,axis = 0)
#    indi = np.argmax(specMatrix,axis = 0)
    
    maxValue = np.amax(specMatrix)
    specMatrix *= 200/maxValue
    output = []
    for timeIndex in range(specMatrix.shape[1]):
        #for timeIndex in range(2):
        column = specMatrix[:,timeIndex]
        """for volume in range(len(column)):
            if column[volume] < 10:
                column[volume] = 0 """
        maxtab , mintab = peakdet(column,1)
        if (len(maxtab)):
            maxtab = maxtab[np.argsort(maxtab[:,1])][::-1]
        else:
            output.append(np.zeros(20))
            continue
        comp = []
        for x in range(5):
            try:
                comp.append(maxtab[x,0])
            except IndexError:
                comp.append(0)
            try:
                comp.append(column[maxtab[x,0]])
            except IndexError:
                comp.append(0)
            try:
                comp.append(column[maxtab[x,0]-1])
            except IndexError:
                comp.append(0)
            try:
                comp.append(column[maxtab[x,0]+1])
            except IndexError:
                comp.append(0)
        output.append(comp)
    np.save("samples"+sample,output)

Num = 18 #Number of all samples
Sam = 3 #Number of samples reserved for test
for name in ["%.2d" % i for i in range(1,Num+1)]:
    preprocessC(name)

arrayA = np.load("samples01.npy")
#Rescale all data to a maxium of 200
for i in ["%.2d" % i for i in range(2,Num+1-Sam)]:
    fileName = "samples"+i+".npy"
    arrayB = np.load(fileName)
    arrayA = np.concatenate((arrayA,arrayB))
np.save("BirdHTraining",arrayA)

for i in ["%.2d" % i for i in range(Num+1-Sam,Num+1)]:
    fileName = "samples"+i+".npy"
    arrayB = np.load(fileName)
    np.save("TestH"+i,arrayB)

