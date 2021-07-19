# import tkinter as tk
# # import colors as c
import numpy as np
from PIL import Image
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()
#
#
# class Game(tk.Frame):
#     def __init__(self):
#         tk.Frame.__init__(self)
#         self.grid()
#         self.master.title('2048')

b = np.random.random((3,3))

c = np.random.random((3,3))

b+=c
print(b**2)