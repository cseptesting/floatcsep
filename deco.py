# class Reg:
#
#     def __init__(self):
#         self.save = []
#
#     def get_path(self, func):
#         def wrapper(arg1):
#             print('a')
#             print(locals())
#             func(arg1)
#             self.save.append(arg1)
#
#         return wrapper
#
#
# class model:
#
#     def __init__(self):
#         self.reg = Reg()
#
#     @reg.get_path
#     def imas(x):
#         print('func', x)
#
#
# imas(1)
# print(reg.save)
# imas(2)
# print(reg.save)
