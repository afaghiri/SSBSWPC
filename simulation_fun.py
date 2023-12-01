def simulation_fun(Fs=2, T=400, corr_fs=.01, corr_amp=.7, simulation_num=1000,
                   winsize=7,signal_Wp=.15, signal_Ws=.2, mod_n=20):
    
    from pathlib import Path
    import numpy as np
    import scipy as sp
    from scipy import signal
    from calculate_SSBSWPC import calculate_SSBSWPC

    # Fs = 2                               # sampling frequency
    # corr_fs = .01                        # corr frequency. corr time series is a cosine with this frequency  
    # simulation_num = 1000                # Number of simulated time series pairs
    # winsize = 7                          # window size used for SSB+SWPC and SWPC 
    # signal_Wp = np.array(signal_Wp)      # passband cutoff frequency for the passband
    # signal_Ws = np.array(signal_Ws)      # stopband cutoff frequency for the passband
    # T = 400                              # length of time series in second
    # mod_n = 20                           # number of freq modulation for SSB.   
    
    Tr = 1/Fs
    tn = np.arange(0,T,Tr)

    # b, a = sp.signal.butter(6, signal_Wp/(Fs/2), btype='lowpass')
    N, Wn = signal.cheb2ord(signal_Wp/(Fs/2), signal_Ws/(Fs/2), gpass=3, gstop=30)
    b, a = signal.cheby2(N, 30, Wn, 'lowpass')

    modfs_vec = np.linspace(0, Fs/2,mod_n+1)[:-1]

    scores_corr_SSB_SWPC = np.zeros([modfs_vec.shape[0], simulation_num, 3])
    scores_RMSE_SSB_SWPC = np.zeros([modfs_vec.shape[0], simulation_num, 3])
    # row_num = simulation_num * modfs_vec.shape[0]
    # scores_corr_SSB_SWPC = np.zeros([row_num, 3])
    # scores_RMSE_SSB_SWPC = np.zeros([row_num, 3])
    cnt = 0
    for modn in np.arange(modfs_vec.shape[0]):
        for simn in np.arange(simulation_num):
            ################################################################################
            # generate random signal with specific connectivity using cholesky decomposition
            Tc_white = np.random.randn(tn.shape[0],2)        
            if signal_Wp <= Fs/2:
                Tc_filt = np.zeros(Tc_white.shape)
                for cn in np.arange(0, Tc_white.shape[1]):    
                    Tc_filt[:,cn] = signal.filtfilt(b, a, Tc_white[:,cn])                
            else:
                Tc_filt = Tc_white               

            corr_tc = corr_amp * np.cos(2*np.pi*corr_fs*tn)   
            Tc = np.zeros(Tc_filt.shape)
            for nn in np.arange(0,corr_tc.shape[0]):
                sigma = np.array([[1, corr_tc[nn]], [corr_tc[nn], 1]])
                # Calculate cholesky decomposition of the covariance matrix and multiply it by the time series signal
                sigma_chdecomp = np.linalg.cholesky(sigma)
                Tc[nn,:] = np.matmul(sigma_chdecomp,Tc_filt[nn,:])
                
            ################################################################################
            # estimate trFC using both SSB+SWPC and classical SWPC (i.e., by puting mod_fs=0)
            SSBSWPC_vec, win_idx = calculate_SSBSWPC(Tc, winsize, Tr, mod_fs = modfs_vec[modn], win_type = 'rect')
            SWPC_vec, win_idx = calculate_SSBSWPC(Tc, winsize, Tr, mod_fs = 0, win_type = 'rect')

            scores_corr_SSB_SWPC[modn, simn, 0] = modfs_vec[modn]
            np.seterr(divide='ignore', invalid='ignore')
            scores_corr_SSB_SWPC[modn, simn, 1] = np.corrcoef(np.squeeze(SSBSWPC_vec),corr_tc[win_idx])[0, 1]
            scores_corr_SSB_SWPC[modn, simn, 2] = np.corrcoef(np.squeeze(SWPC_vec),corr_tc[win_idx])[0, 1]
            np.seterr(divide='warn', invalid='warn')
            
            scores_RMSE_SSB_SWPC[modn, simn, 0] = modfs_vec[modn]
            scores_RMSE_SSB_SWPC[modn, simn, 1] = np.mean((np.squeeze(SSBSWPC_vec) - corr_tc[win_idx])**2)**.5
            scores_RMSE_SSB_SWPC[modn, simn, 2] = np.mean((np.squeeze(SWPC_vec) - corr_tc[win_idx])**2)**.5
            cnt = cnt + 1
        print(f"progress {(modn+1)/modfs_vec.shape[0]*100:.2f}%", end="\r")
        
    root_dir = 'sim_npy_files/'
    file_name = f'fs_{corr_fs:.3f}__amp_{corr_amp:.1f}__Wp_{signal_Wp:.2f}__winsize_{winsize:03d}'
    
    file_name_corr = file_name + '_corr.npy'
    file_name_RMSE = file_name + '_RMSE.npy'
    print(file_name_corr)
    print(file_name_RMSE)
    
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    np.save(root_dir + file_name_corr, scores_corr_SSB_SWPC)
    np.save(root_dir + file_name_RMSE, scores_RMSE_SSB_SWPC)