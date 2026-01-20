# Cluster 11

def test_tpm():
    tpm = TPM(2000)
    for i in range(20):
        tpm.wait(silent=False, token_count=150)
        print(i)

