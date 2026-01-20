# Cluster 18

class AGasTank:

    def __init__(self, amount: int):
        self.gas = amount
        return

    def Set(self, amount: int):
        self.gas = amount
        return

    def Consume(self, resourceType: str, amount) -> int:
        if self.gas - amount < 0:
            raise AExceptionOutofGas()
        self.gas -= amount
        return self.gas

    def Charge(self, amount: int) -> int:
        self.gas += amount
        return self.gas

