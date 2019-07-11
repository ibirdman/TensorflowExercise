import argparse
import os
import sys
from six.moves import urllib
import jieba

seg_list = jieba.cut("猫咪漂流在江河上", cut_all=False)
print("【精确模式】：" + "/".join(seg_list))


seg_list = jieba.cut_for_search("他毕业于上海交通大学机电系，后来在一机部上海电器科学研究所工作")
print("【搜索引擎模式】：" + "/ ".join(seg_list))