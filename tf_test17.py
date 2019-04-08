from functools import reduce
import numpy as np

def fn(x, y):
   return max(x, y)

result = reduce(fn, [1, 3, 5, 7, 9])
print(result)

url = "https://www.cnblogs.com/" \
      "dinghanhua" \
      "/p/9900700.html"
print(url)