# -*- coding: utf-8 -*-
"""Implements an acoustic path impulse response based on https://www.mathworks.com/help/audio/ug/acoustic-echo-cancellation-aec.html """
import numpy as np
from scipy.signal import cheby2, lfilter

def generate_echo_response(sampling_rate: int, reverberation_time: float, response_length:int, total_power:float, delay:float)->np.ndarray:
    h = np.zeros(response_length)
    nR = int(sampling_rate*reverberation_time)
    n0 = int(sampling_rate*delay)
    
    [b,a] = cheby2(N=4, rs=20, Wn=[0.1, 0.7], btype='bandpass', analog=False, output='ba')
    
    htemp = np.random.normal (0, 
                              np.exp(-6*np.log(10)/nR * np.arange( 0, response_length ) )
                             )
    htemp = lfilter(b,a,htemp)
    htemp = htemp/np.linalg.norm(htemp,2)
    h[n0:] = htemp[:(len(htemp)-n0)]
    return h