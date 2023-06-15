import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as spwin
import scipy as sp
from scipy.fft import fft, fftfreq
from numpy.matlib import repmat


def calculate_SSBSWPC(Tc, winsize, Tr=2, mod_fs=0, win_type = 'rect', plot_flag = 0):
    """calculate SSB+SWPC
    calculate pairwise Sliding window Pearson correaltion (SWPC) using 
    single side band modulation (SSB). Using SSB+SWPC instead of classic 
    SWPC, allows us to select shorter window sizes without removing 
    important low-frequency information of the inputs.
    
    Parameters
    ----------
    Tc : numpy.array
        input signals (sample number, component number).
    winsize : int
        window size for the algorithm
    Tr : float, optional
        sampling interval in time. This value is important for visulaization
        , by default 2
    mod_fs : int, optional
        modulation frequency in Hz, by default 0. If equal to 0, the same as 
        SWPC
    win_type : str, optional
        ype of window. Can be 'rect' for rectangular, 'gauss' for gaussian, 
        and 'tukey' for tukey (tapered cosine) window., by default 'rect'
    plot_flag : int, optional
        if 1, generate and plot informative figures that aid in assessing the accuracy or
        validity of the chosen parameters, by default 0

    Returns
    -------
    SSBSWPC_vec : numpy.array
        estimated time resolved sample Pearson correaltion, i.e., time resolved connectivity.
        (window number, numper of pairs). window number is equal to Tc.shape[0]-winsize+1
        while numper of pairs is equal to Tc.shape[1]*(Tc.shape[1]-1)/2. SSBSWPC_vec is 
        vectorized version of the connectivity matrix
    win_idx : numpy.array
        window center index as a vector (window number,). Can be usefull for matching the 
        SSBSWPC_vec with any time series in the same temporal space as Tc
    
    Raises
    ------
    ValueError
        _description_
    """
    Fs = 1/Tr
    match win_type:
        case 'rect':
            window_tc = spwin.boxcar(winsize)
        case 'gauss':
            win_alpha = 2.5
            win_std = (winsize-1)/(2*win_alpha)
            window_tc = spwin.gaussian(winsize, std=win_std)
        case 'tukey':
            window_tc = spwin.tukey(winsize, alpha=0.5)
        case _:
            raise ValueError("win_type should be 'rect', 'gauss', or 'tukey'.")
            # print("win_type should be 'rect', 'gauss', or 'tukey'.")
            # return -1
    window_tc = window_tc / sum(window_tc)
            
    tt = np.arange(0, Tc.shape[0]*Tr, Tr)
    modulation_tc = np.exp(1j*2*np.pi*tt*mod_fs)
    modulation_tc_mat = modulation_tc[:, np.newaxis] * np.ones((1, Tc.shape[1]))

    Tca = sp.signal.hilbert(Tc, axis=0)
    Tcam = np.real(Tca * modulation_tc_mat)

    window_num = Tc.shape[0]-winsize+1
    SSBSWPC_mat = np.zeros((window_num, Tc.shape[1], Tc.shape[1]))
    win_idx = np.zeros(window_num)
    for w_start in np.arange(0,window_num):
        widx = np.arange(w_start, w_start+winsize)
        Tcam_windowed = Tcam[widx, :]
        
        # calculate weighted correlation for the window time series
        temp_mean = np.matmul(np.transpose(window_tc[:,np.newaxis]), Tcam_windowed)
        temp_mean = repmat(temp_mean, Tcam_windowed.shape[0], 1)
        temp_demean = Tcam_windowed - temp_mean
        window_tc_mat = repmat(window_tc[:, np.newaxis], 1, Tcam_windowed.shape[1])
        temp = np.matmul(np.transpose(temp_demean), temp_demean*window_tc_mat) 
        temp = 0.5 * (temp + np.transpose(temp))
        R = np.diag(temp)
        SSBSWPC_mat[w_start, :, :] = temp / np.sqrt(np.matmul(R[:, np.newaxis], np.transpose(R[:, np.newaxis])))
        win_idx[w_start] = int(np.median(widx))

    vec_num = SSBSWPC_mat.shape[1]*(SSBSWPC_mat.shape[1]-1)/2
    SSBSWPC_vec = np.zeros([SSBSWPC_mat.shape[0], int(vec_num)])
    cnt = 0
    for ii in np.arange(0, SSBSWPC_mat.shape[1]):
        for jj in np.arange(ii+1, SSBSWPC_mat.shape[2]):
            SSBSWPC_vec[:,cnt] = SSBSWPC_mat[:, ii, jj]
            cnt = cnt + 1
        
    if plot_flag==1:
        Tc_fft = fft(Tc, axis=0)
        Tc_fft = Tc_fft[0:Tc.shape[0]//2, :]
        
        Tcam_fft = fft(Tcam, axis=0)
        Tcam_fft = Tcam_fft[0:Tcam_fft.shape[0]//2, :]
        
        Tcam_mult = Tcam[:,np.newaxis,:] * Tcam[:,:,np.newaxis]
        tmp_sz = Tcam_mult.shape
        Tcam_mult_vec = np.reshape(Tcam_mult, (tmp_sz[0], tmp_sz[1]*tmp_sz[2]))
        Tcam_mult_fft = fft(Tcam_mult_vec, axis=0)
        Tcam_mult_fft = Tcam_mult_fft[0:Tc.shape[0]//2, :]
        
        delta_tc = np.zeros(window_tc.shape)
        delta_tc[int((winsize+1)/2)] = 1
        window_fft_hpf = fft(delta_tc - window_tc, n=Tc.shape[0], axis=0)    
        window_fft_hpf = window_fft_hpf[0:Tc.shape[0]//2]
        
        window_fft_lpf = fft(window_tc, n=Tc.shape[0], axis=0)    
        window_fft_lpf = window_fft_lpf[0:Tc.shape[0]//2]
        
        freq_val = fftfreq(Tc.shape[0], Fs)[:Tc.shape[0]//2]
        
        fig, axes = plt.subplots(2, 1, figsize=[4, 5])
        
        tmp_fft = np.mean(np.abs(Tc_fft), axis=1)
        tmp_fft = tmp_fft / np.max(tmp_fft)
        axes[0].plot(freq_val, tmp_fft, label='Tc')        
        tmp_fft = np.mean(np.abs(Tcam_fft), axis=1)
        tmp_fft = tmp_fft / np.max(tmp_fft)
        axes[0].plot(freq_val, tmp_fft, label='modulated Tc')        
        tmp_fft = np.abs(window_fft_hpf)
        tmp_fft = tmp_fft / np.max(tmp_fft)
        axes[0].plot(freq_val, tmp_fft, 'k--', label='HPF transfer function')
        axes[0].set_ylim([-.01, 1.4])
        axes[0].set_ylabel('Normalized amplitude')
        axes[0].set_xlabel('Freq (Hz)')
        axes[0].legend(loc='upper right')
        
        tmp_fft = np.mean(np.abs(Tcam_mult_fft), axis=1)
        tmp_fft = tmp_fft / np.max(tmp_fft)
        print(tmp_fft.shape)
        print(freq_val.shape)
        axes[1].plot(freq_val, tmp_fft, label='Conn')
        tmp_fft = np.abs(window_fft_lpf)
        tmp_fft = tmp_fft / np.max(window_fft_lpf)
        axes[1].plot(freq_val, tmp_fft, 'k--', label='LPF transfer function')
        axes[1].set_ylim([-.01, 1.4])
        axes[1].set_ylabel('Normalized amplitude')
        axes[1].set_xlabel('Freq (Hz)')
        axes[1].legend(loc='upper right')
        
    win_idx = np.int64(win_idx)
    return SSBSWPC_vec, win_idx

    