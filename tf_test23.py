import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#设置字体样式
mpl.rcParams['font.family']='sans-serif'
mpl.rcParams['font.sans-serif']=[u'SimHei']
title = 'Hello'
fig = plt.figure(title, figsize=(8, 7))
fig.tight_layout()
ax = fig.add_subplot(111)
data = [3,4,5,7,3]
labels = ['a','b','c','d','e']
explodes =[0 for x in data]
explodes[0] =0.015
ax.pie(data, labels= labels, radius=0.8, #data 是数据，labels 是标签，radius 是饼图半径
       explode=explodes, #explodes 为0 代表不偏离圆心， 不为零则代表偏离圆心的距离
       autopct='%1.1f%%', #显示所占比例，百分数
       pctdistance = 0.5,
       labeldistance=0.7,  # a,b,c,d 到圆心的距离
       textprops={'fontsize': 20, 'color': 'black'}) # 标签和比例的格式
plt.axis('equal') # 正圆
plt.legend( loc = 'upper right',bbox_to_anchor=(1.1, 1.05), fontsize=14, borderaxespad=0.3)
# loc =  'upper right' 位于右上角
# bbox_to_anchor=[0.5, 0.5] # 外边距 上边 右边
# ncol=2 分两列
# borderaxespad = 0.3图例的内边距
plt.suptitle(title+'pie', fontsize=20)
#plt.savefig(filepath+'\name.png',dpi=120,bbox_inches='tight') #可通过这个方法保存可视化的图片
plt.show()
plt.close()
plt.savefig('bingtu.png',dpi=120,bbox_inches='tight')

'''(x, explode=None, labels=None, colors=None, autopct=None,
pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=None,
radius=None, counterclock=True, wedgeprops=None, textprops=None,
center=(0, 0), frame=False, rotatelabels=False, hold=None, data=None)'''
# 参数说明：
# x：数组。输入的数据用于创建一个饼图。
# explode：数组，可选参数，默认为None。
#         如果不是None，是一个长度与x相同长度的数组，用来指定每部分的偏移量。
#         例如：explode=[0,0,0.2,0,0]，第二个饼块被拖出。
# labels：列表，可选参数，默认为：None。
#        一个字符串序列作为每个饼块的标记。
# colors：数组，可选参数，默认为：None。
#       用来标注每块饼图的matplotlib颜色参数序列。
#       如果为None，将使用当前活动环的颜色。
# autopct：默认是None，字符串或函数，可选参数。
#         如果不是None，是一个字符串或函数用带有数值饼图标注。
# pctdistance：浮点数，可选参数，默认值：0.6。
#           每个饼切片的中心和通过autopct生成的文本开始之间的比例。
#           如果autopct是None，被忽略。
# shadow：布尔值，可选参数，默认值：False。
#         在饼图下面画一个阴影。
# labeldistance：浮点数，可选参数，默认值：1.1。
#             被画饼标记的直径。
# startangle：浮点类型，可选参数，默认：None。
#           如果不是None，从x轴逆时针旋转饼图的开始角度。
# radius：浮点类型，可选参数，默认为：None。
#       饼图的半径，如果半径是None，将被设置成1。
# counterclock：布尔值，可选参数，默认为：None。
#             指定指针方向，顺时针或者逆时针。
# wedgeprops：字典类型，可选参数，默认值：None。
#             参数字典传递给wedge对象用来画一个饼图。
#             例如：wedgeprops={'linewidth':3}设置wedge线宽为3。
# textprops：字典类型，可选参数，默认值为：None。
#           传递给text对象的字典参数。
# center：浮点类型的列表，可选参数，默认值：(0,0)。
#       图标中心位置。
# frame：布尔类型，可选参数，默认值：False。
#       如果是true，绘制带有表的轴框架。
# rotatelabels：布尔类型，可选参数，默认为：False。
#           如果为True，旋转每个label到指定的角度。
# 返回值：
# patches：列表。matplotlib.patches.Wedge实例列表。
# text：列表。matplotlib.text.Text实例label的列表。
# autotexts：列表。A是数字标签的Text实例列表。
#           仅当参数autopct不为None时才返回。
# '''