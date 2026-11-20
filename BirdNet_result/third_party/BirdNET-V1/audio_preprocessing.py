"""Spectrogram functions copied from BirdNET-V1/utils/audio.py.
Original BirdNET team; CC BY-NC-SA 4.0. See BirdNET-V1/LICENSE.
Only imports and unrelated functions were removed.
"""
import numpy as np
import scipy.signal
CACHE = {}

def buildBandpassFilter(rate, fmin, fmax, order=4):

    global CACHE

    fname = 'bandpass_' + str(rate) + '_' + str(fmin) + '_' + str(fmax)
    if not fname in CACHE:
        wn = np.array([fmin, fmax]) / (rate / 2.0)
        filter_sos = scipy.signal.butter(order, wn, btype='bandpass', output='sos')

        # Save to cache
        CACHE[fname] = filter_sos

    return CACHE[fname]

def applyBandpassFilter(sig, rate, fmin, fmax):

    # Build filter or load from cache
    filter_sos = buildBandpassFilter(rate, fmin, fmax)

    return scipy.signal.sosfiltfilt(filter_sos, sig)

def pcen(spec, rate, hop_length, gain=0.8, bias=10, power=0.25, t=0.060, eps=1e-6):
    s = 1 - np.exp(- float(hop_length) / (t * rate))
    M = scipy.signal.lfilter([s], [1, s - 1], spec)
    smooth = (eps + M)**(-gain)
    return (spec * smooth + bias)**power - bias**power

def get_mel_filterbanks(num_banks, fmin, fmax, f_vec, dtype=np.float32):
    '''
    An arguably better version of librosa's melfilterbanks wherein issues with "hard snapping" are avoided. Works with
    an existing vector of frequency bins, as returned from signal.spectrogram(), instead of recalculating them and
    flooring down the bin indices.
    '''

    global CACHE

    # Filterbank already in cache?
    fname = 'mel_' + str(num_banks) + '_' + str(fmin) + '_' + str(fmax)
    if not fname in CACHE:

        # Break frequency and scaling factor
        A = 4581.0
        f_break = 1750.0

        # Convert Hz to mel
        freq_extents_mel = A * np.log10(1 + np.asarray([fmin, fmax], dtype=dtype) / f_break)

        # Compute points evenly spaced in mels
        melpoints = np.linspace(freq_extents_mel[0], freq_extents_mel[1], num_banks + 2, dtype=dtype)

        # Convert mels to Hz
        banks_ends = (f_break * (10 ** (melpoints / A) - 1))

        filterbank = np.zeros([len(f_vec), num_banks], dtype=dtype)
        for bank_idx in range(1, num_banks+1):
            # Points in the first half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx - 1], f_vec <= banks_ends[bank_idx])
            filterbank[mask, bank_idx-1] = (f_vec[mask] - banks_ends[bank_idx - 1]) / \
                (banks_ends[bank_idx] - banks_ends[bank_idx - 1])

            # Points in the second half of the triangle
            mask = np.logical_and(f_vec >= banks_ends[bank_idx], f_vec <= banks_ends[bank_idx+1])
            filterbank[mask, bank_idx-1] = (banks_ends[bank_idx + 1] - f_vec[mask]) / \
                (banks_ends[bank_idx + 1] - banks_ends[bank_idx])

        # Scale and normalize, so that all the triangles do not have same height and the gain gets adjusted appropriately.
        temp = filterbank.sum(axis=0)
        non_zero_mask = temp > 0
        filterbank[:, non_zero_mask] /= np.expand_dims(temp[non_zero_mask], 0)

        # Save to cache
        CACHE[fname] = (filterbank, banks_ends[1:-1])

    return CACHE[fname][0], CACHE[fname][1]

def spectrogram(sig, rate, shape=(64, 512), win_len=512, fmin=150, fmax=15000, frequency_scale='mel', magnitude_scale='nonlinear', bandpass=True, decompose=False):

    # Compute overlap
    hop_len = int(len(sig) / (shape[1] - 1))
    win_overlap = win_len - hop_len + 2
    #print 'WIN_LEN:', win_len, 'HOP_LEN:', hop_len, 'OVERLAP:', win_overlap

    # Adjust N_FFT?
    if frequency_scale == 'mel':
        n_fft = win_len
    else:
        n_fft = shape[1] * 2

    # Bandpass filter?
    if bandpass:
        sig = applyBandpassFilter(sig, rate, fmin, fmax)

    # Compute spectrogram
    f, t, spec = scipy.signal.spectrogram(sig,
                                          fs=rate,
                                          window=scipy.signal.windows.hann(win_len),
                                          nperseg=win_len,
                                          noverlap=win_overlap,
                                          nfft=n_fft,
                                          detrend=False,
                                          mode='magnitude')

    # Scale frequency?
    if frequency_scale == 'mel':

        # Determine the indices of where to clip the spec
        valid_f_idx_start = f.searchsorted(fmin, side='left')
        valid_f_idx_end = f.searchsorted(fmax, side='right') - 1

        # Get mel filter banks
        mel_filterbank, mel_f = get_mel_filterbanks(shape[0], fmin, fmax, f, dtype=spec.dtype)

        # Clip to non-zero range so that unnecessary multiplications can be avoided
        mel_filterbank = mel_filterbank[valid_f_idx_start:(valid_f_idx_end + 1), :]

        # Clip the spec representation and apply the mel filterbank.
        # Due to the nature of np.dot(), the spec needs to be transposed prior, and reverted after
        spec = np.transpose(spec[valid_f_idx_start:(valid_f_idx_end + 1), :], [1, 0])
        spec = np.dot(spec, mel_filterbank)
        spec = np.transpose(spec, [1, 0])

    # Magnitude transformation
    if magnitude_scale == 'pcen':

        # Convert scale using per-channel energy normalization as proposed by Wang et al., 2017
        # We adjust the parameters for bird voice recognition based on Lostanlen, 2019
        spec = pcen(spec, rate, hop_len)

    elif magnitude_scale == 'log':

        # Convert power spec to dB scale (compute dB relative to peak power)
        spec = spec ** 2
        spec = 10.0 * np.log10(np.maximum(1e-10, spec) / np.max(spec))
        spec = np.maximum(spec, spec.max() - 100) # top_db = 100

    elif magnitude_scale == 'nonlinear':

        # Convert magnitudes using nonlinearity as proposed by SchlÃ¼ter, 2018
        a = -1.2 # Higher values yield better noise suppression
        s = 1.0 / (1.0 + np.exp(-a))
        spec = spec ** s

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    spec = spec[::-1, ...]

    # Trim to desired shape if too large
    spec = spec[:shape[0], :shape[1]]

    # Normalize values between 0 and 1
    spec -= spec.min()
    if not spec.max() == 0:
        spec /= spec.max()
    else:
        spec = np.clip(spec, 0, 1)

    return spec
