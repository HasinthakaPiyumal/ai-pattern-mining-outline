# Cluster 7

def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for k, v in d.items():
        logkv(k, v)

def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    Logger.CURRENT.logkv(key, val)

