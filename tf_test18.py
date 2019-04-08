from functools import reduce
import numpy as np


def char2num(s):
   return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]


def str2int(s):
   return reduce(lambda x, y: x * 10 + y, map(char2num, s))


def return_float(s):
   base = 0.1
   base_minus = 10
   stra = "1"
   for i in range(s - 1):
      stra = stra + '0'
   return base / str2int(stra)


def split_float(arg, l_or_r):
   if l_or_r == "l":
      return arg.split('.')[0]
   else:
      return arg.split('.')[1]


def str2float(s):
   left = split_float(s, "l")
   right = split_float(s, "r")
   final_left = str2int(left)
   r_float = 10**len(right)
   final_right = str2int(right) / r_float
   return final_left + final_right


b = '123.456'
c = str2float(b)
print(c)