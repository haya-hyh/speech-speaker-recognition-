# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------
import numpy as np
from scipy.signal import lfilter,hamming
from  scipy.fftpack import fft
from lab1_tools import *

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    mspecs,_ = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
def enframe(samples, winlen, winshift, samplingrate=20000):
    winlen_samples = int(winlen * samplingrate)
    winshift_samples = int(winshift*samplingrate)
    n_f = int((len(samples) - winlen_samples) / winshift_samples) + 1
    frames = np.zeros((n_f, winlen_samples))
    for i in range(n_f):
        start = i * winshift_samples
        end = start + winlen_samples
        frames[i, :] = samples[start:end]
    return frames



    
def preemp(input, p=0.97):
    b = [1, -p]
    a = [1] 
    output = np.zeros_like(input)
    for i in range(input.shape[0]):
        output[i] = lfilter(b, a, input[i])
    return output


def windowing(input):
    _, M = input.shape
    wind = hamming(M, sym=False)
    return input * wind

def powerSpectrum(input, nfft):
    fft_input = fft(input,nfft)
    power_spec = np.abs(fft_input)**2
    return power_spec


def logMelSpectrum(input, samplingrate):
    n_frames, nfft = input.shape
    fbank = trfbank(samplingrate, nfft)
    mel_spec = np.dot(input, fbank.T)
    log_mel = np.log(mel_spec)
    return log_mel, fbank

from scipy.fftpack.realtransforms import dct
def cepstrum(input, nceps):
    return dct(input)[:,: nceps]#－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

def compute_dist(x, y, winlen, winshift):
    x_mfcc = mfcc(x, winlen, winshift)
    y_mfcc = mfcc(y, winlen, winshift)
    N = x_mfcc.shape[0]
    M = y_mfcc.shape[0] 
    distance = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            distance[i, j] = np.linalg.norm(x_mfcc[i] - y_mfcc[j])
    return distance

# attention here dist is not function
def dtw(dist):
    N,M = dist.shape
    LD = dist
    AD = np.zeros((N, M))
    AD[0, 0] = LD[0, 0]

    # boundary
    for i in range(1, N):
        AD[i, 0] = LD[i, 0] + AD[i-1, 0]
    for j in range(1, M):
        AD[0, j] = LD[0, j] + AD[0, j-1]

    for i in range(1, N):
        for j in range(1, M):
            AD[i, j] = LD[i, j] + min(AD[i-1, j], AD[i, j-1], AD[i-1, j-1])
    
    #path findingggggg
    path = [(N-1, M-1)]
    i, j = N-1, M-1
    while i > 0 and j > 0:
        direction = np.argmin([AD[i-1, j-1], AD[i-1, j], AD[i, j-1]])
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
        path.append((i, j))

    while(i>0 or j>0):
        if i > 0:
            i -= 1
            path.append((i, j))
        else:
            j -= 1
            path.append((i, j))

    
    d = AD[N-1, M-1] / (N + M) # may delete
    
    return d, AD, path
