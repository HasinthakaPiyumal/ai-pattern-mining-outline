# Cluster 14

def spectrum_power(frames, NFFT):
    """Calculate power spectrum for every frame after FFT.
    Args:
        frames: 2-D frames array calculated by audio2frame(...).
        NFFT:FFT size
    Returns:
        Power spectrum: PS = magnitude^2/NFFT
    """
    return 1.0 / NFFT * numpy.square(spectrum_magnitude(frames, NFFT))

def spectrum_magnitude(frames, NFFT):
    """Apply FFT and Calculate magnitude of the spectrum.
    Args:
        frames: 2-D frames array calculated by audio2frame(...).
        NFFT:FFT size.
    Returns:
        Return magnitude of the spectrum after FFT, with shape (frames_num, NFFT).
    """
    complex_spectrum = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spectrum)

def log_spectrum_power(frames, NFFT, norm=1):
    """Calculate log power spectrum.
    Args:
        frames:2-D frames array calculated by audio2frame(...)
        NFFT：FFT size
        norm: Norm.
    """
    spec_power = spectrum_power(frames, NFFT)
    spec_power[spec_power < 1e-30] = 1e-30
    log_spec_power = 10 * numpy.log10(spec_power)
    if norm:
        return log_spec_power - numpy.max(log_spec_power)
    else:
        return log_spec_power

def fbank(signal, samplerate=16000, win_length=0.025, win_step=0.01, filters_num=26, NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97):
    """Perform pre-emphasis -> framing -> get magnitude -> FFT -> Mel Filtering.
    Args:
        signal: 1-D numpy array.
        samplerate: Sampling rate. Defaulted to 16KHz.
        win_length: Window length. Defaulted to 0.025, which is 25ms/frame.
        win_step: Interval between the start points of adjacent frames.
            Defaulted to 0.01, which is 10ms.
        cep_num: Numbers of cepstral coefficients. Defaulted to 13.
        filters_num: Numbers of filters. Defaulted to 26.
        NFFT: Size of FFT. Defaulted to 512.
        low_freq: Lowest frequency.
        high_freq: Highest frequency.
        pre_emphasis_coeff: Coefficient for pre-emphasis. Pre-emphasis increase
            the energy of signal at higher frequency. Defaulted to 0.97.
    Returns:
        feat: Features.
        energy: Energy.
    """
    high_freq = high_freq or samplerate / 2
    signal = pre_emphasis(signal, pre_emphasis_coeff)
    frames = audio2frame(signal, win_length * samplerate, win_step * samplerate)
    spec_power = spectrum_power(frames, NFFT)
    energy = numpy.sum(spec_power, 1)
    energy = numpy.where(energy == 0, numpy.finfo(float).eps, energy)
    fb = get_filter_banks(filters_num, NFFT, samplerate, low_freq, high_freq)
    feat = numpy.dot(spec_power, fb.T)
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)
    return (feat, energy)

def pre_emphasis(signal, coefficient=0.95):
    """Pre-emphasis.
    Args:
        signal: 1-D numpy array.
        coefficient:Coefficient for pre-emphasis. Defauted to 0.95.
    Returns:
        pre-emphasis signal.
    """
    return numpy.append(signal[0], signal[1:] - coefficient * signal[:-1])

def audio2frame(signal, frame_length, frame_step, winfunc=lambda x: numpy.ones((x,))):
    """ Framing audio signal. Uses numbers of samples as unit.

    Args:
    signal: 1-D numpy array.
	frame_length: In this situation, frame_length=samplerate*win_length, since we
        use numbers of samples as unit.
    frame_step:In this situation, frame_step=samplerate*win_step,
        representing the number of samples between the start point of adjacent frames.
	winfunc:lambda function, to generate a vector with shape (x,) filled with ones.

    Returns:
        frames*win: 2-D numpy array with shape (frames_num, frame_length).
    """
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    if signal_length <= frame_length:
        frames_num = 1
    else:
        frames_num = 1 + int(math.ceil((1.0 * signal_length - frame_length) / frame_step))
    pad_length = int((frames_num - 1) * frame_step + frame_length)
    zeros = numpy.zeros((pad_length - signal_length,))
    pad_signal = numpy.concatenate((signal, zeros))
    indices = numpy.tile(numpy.arange(0, frame_length), (frames_num, 1)) + numpy.tile(numpy.arange(0, frames_num * frame_step, frame_step), (frame_length, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    frames = pad_signal[indices]
    win = numpy.tile(winfunc(frame_length), (frames_num, 1))
    return frames * win

def log_fbank(signal, samplerate=16000, win_length=0.025, win_step=0.01, filters_num=26, NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97):
    """Calculate log of features.
    """
    feat, energy = fbank(signal, samplerate, win_length, win_step, filters_num, NFFT, low_freq, high_freq, pre_emphasis_coeff)
    return numpy.log(feat)

def ssc(signal, samplerate=16000, win_length=0.025, win_step=0.01, filters_num=26, NFFT=512, low_freq=0, high_freq=None, pre_emphasis_coeff=0.97):
    """
    待补充
    """
    high_freq = high_freq or samplerate / 2
    signal = pre_emphasis(signal, pre_emphasis_coeff)
    frames = audio2frame(signal, win_length * samplerate, win_step * samplerate)
    spec_power = spectrum_power(frames, NFFT)
    spec_power = numpy.where(spec_power == 0, numpy.finfo(float).eps, spec_power)
    fb = get_filter_banks(filters_num, NFFT, samplerate, low_freq, high_freq)
    feat = numpy.dot(spec_power, fb.T)
    R = numpy.tile(numpy.linspace(1, samplerate / 2, numpy.size(spec_power, 1)), (numpy.size(spec_power, 0), 1))
    return numpy.dot(spec_power * R, fb.T) / feat

