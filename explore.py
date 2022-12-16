from functools import singledispatch


class A:

    def __init__(self):
        self.b = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
        self.c = 1


a = A()

c = getattr(a, 'cc', 'hola')
print(c)
