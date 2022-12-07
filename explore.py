from functools import singledispatch


class Argh:

    def __init__(self):
        self.x = 1

    @property
    def y(self, a=1):
        return 2 * self.x * a


a = Argh()
a.y(2)
print(a.y(2))
