from scipy import signal
from scipy import stats
from scipy import integrate
from scipy import fft
import numpy as np


class Track:
    def __init__(self, freq, length):
        self.centre_freq=freq
        self.peak_present = [];
        for i in range(length):
            #-1000 is a flag for non active track
            self.peak_present.append(-1000)

    def update_track(self,position, value):
        self.peak_present[position] = value
#frequency tracking
def create_tracks(data, sampling_rate):
    # create blocks
    blocks = [data[x:x + 2048] for x in range(0, len(data), 2048)]
    sr= sampling_rate
    window = 'hamming'
    tracks = []
    for block in blocks:
        #fft on block
        n = len(block)
        fft_block = fft(block)
        freqs = np.fft.fftfreq(n)
        # Find the peak in the coefficients
        idx = np.argmax(np.abs(fft_block))
        peak_indexes = np.abs(fft_block).find_peaks()
        new_peaks=[]
        for index in peak_indexes:
            peak = freqs[index]
            peak_in_hz = abs(peak * sr)
            new_peaks.append(peak_in_hz)

        # each track has a bandwidth of 100Hz
        # take the freq, make a copy, integer divide by 100 and multiply by 100
        # gets the freq block
        for peak in new_peaks:
            peak_group = peak // 100 * 100
            # Flag if we need to create a new track
            existing_track = False
            # this for loop will determine if we need to update or create a new track
            for track in tracks:
                # if a track already exists, note flag to not create a new track
                # update existing track
                if peak_group == track.centre_freq:
                    existing_track = True
                    track.update_track(i, peak - peak_group)
            # create new track
            if not existing_track:
                temp_track = Track(peak_group, len(blocks))
                temp_track.update_track(i, peak - peak_group)
                tracks.append(temp_track)

    return tracks
#pitch generation
def pitch_curve_generation(tracks):
    p = []
    tracks = np.array(tracks)
    #generate Diagonal of R and N
    #is the same size as tracks
    M = np.zeros((len(tracks),len(tracks[0])))
    R_list = []
    N_list = []
    #generate M and N_list
    for t in len(tracks):
        active_track_length=0
        for i in len(t[t]):
            #if it doesnt equal the flag then the track is active
            if(tracks[t][i]!=-1000):
                M[t][i]=1
                active_track_length=active_track_length+1
        N_list.append(active_track_length)
    #generate R_list
    for j in len(tracks[0]):
        num_active_tracks=0
        for k in len(track):
            if(tracks[j][k]!=-1000):
                num_active_tracks=num_active_tracks+1
        R_list.append(num_active_tracks)

    #generate the diagonal matrices
    R_diagonal= np.diag(R_list)
    N_diagonal= np.diag(N_list)
    #generate column vectors
    vect_n = np.ones(len(tracks[0]),1)
    vect_r_max = np.ones(len(tracks),1)
    #generate diagonal matrix with gaussian
    c_p = np.diag(np.diag(np.random.normal(0,1,len(track[0]))))
    #sigma^2
    sigma = np.random.normal(0, 1, 1)

    #equation
    inverse_part = np.linalg.inv(R_diagonal - np.matmul(np.matmul(M.transpose(), np.linalg.inv(N_diagonal)), M) + sigma*sigma*np.linalg.inv(c_p))
    second_part = np.matmul(tracks.transpose(), vect_r_max.transpose())-np.matmul(np.matmul(M.transpose(),np.linalg.inv(N_diagonal)), np.matmul(tracks,vect_n.transpose()))
    p = np.matmul(inverse_part, second_part)

    #equation
    f_0 = np.linalg.inv(N_diagonal)
    print(p)
    return p
#resampling
def resampling(p,sr,degraded_signal):
    resampled_signal = []
    #number of samples to consider is 2M+1
    M=256
    #sampling rate
    T=sr
    tau = integrate.cumtrapz(p)
    for n in len(degraded_signal):
        #generate the variables needed
        T_o = T/p[n]
        alpha = min([1, T_o / T])
        tau_n= tau[n]
        #use the 2M+1 closest samples
        #account for edge cases of starting and ending
        if n-M/2 < 0:
            start = 0
            end = n+M/2
        elif n+M/2 > len(degraded_signal):
            start = n-M/2
            end = len(degraded_signal)
        else:
            start = n-M/2
            end = n+M/2
        sample = 0
        for m in range(start,end):
            # windowing function: (m*T_o-tau)
            w = signal.windows.hamming(m * T_o - tau_n)
            sinc_val = np.sinc(alpha * (m * T_o - tau_n))
            x_w = degraded_signal[(n - m) * T_o]
            #this might need to be changed
            result = w*sinc_val*x_w
            sample = sample + result
        resampled_signal.append(sample)
    return resampled_signal

if __name__ == '__main__':
    #create frequency tracks
    freq_tracks = create_tracks("test.wav")
    #extract into matrix
    final_freq_tracks=[]
    for track in freq_tracks:
        freq = []
        for i in range(len(track.peak_present)):
            freq.append(track.peak_present[i])
        final_freq_tracks.append(freq)
    #pitch generation
    p = pitch_curve_generation(final_freq_tracks)
    #resample
    signal = resampling(p)