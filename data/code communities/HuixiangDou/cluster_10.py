# Cluster 10

def test_rpm():
    rpm = RPM(30)
    for i in range(40):
        rpm.wait()
        print(i)
    time.sleep(5)
    for i in range(40):
        rpm.wait()
        print(i)

