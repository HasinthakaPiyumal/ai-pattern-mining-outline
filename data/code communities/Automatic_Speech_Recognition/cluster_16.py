# Cluster 16

def get_filter_banks(filters_num=20, NFFT=512, samplerate=16000, low_freq=0, high_freq=None):
    """Calculate Mel filter banks.
    Args:
        filters_num: Numbers of Mel filters.
        NFFT:FFT size. Defaulted to 512.
        samplerate: Sampling rate. Defaulted to 16KHz.
        low_freq: Lowest frequency.
        high_freq: Highest frequency.
    """
    low_mel = hz2mel(low_freq)
    high_mel = hz2mel(high_freq)
    mel_points = numpy.linspace(low_mel, high_mel, filters_num + 2)
    hz_points = mel2hz(mel_points)
    bin = numpy.floor((NFFT + 1) * hz_points / samplerate)
    fbank = numpy.zeros([filters_num, NFFT // 2 + 1])
    for j in xrange(0, filters_num):
        for i in xrange(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in xrange(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank

def hz2mel(hz):
    """Convert frequency to Mel frequency.
    Args:
        hz: Frequency.
    Returns:
        Mel frequency.
    """
    return 2595 * numpy.log10(1 + hz / 700.0)

def mel2hz(mel):
    """Convert Mel frequency to frequency.
    Args:
        mel:Mel frequency
    Returns:
        Frequency.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)

