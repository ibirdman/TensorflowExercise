import tensorflow as tf
import numpy as np

import types

class Person(object):
    def __init__(self, newName, newAge):
        self.name = newName
        self.age = newAge

    def eat(self):
        print("--%s-正在吃---" % self.name)

def run(self):
    print("--%s-正在跑---" % self.name)

@classmethod
def classRun(cls):
    print("---正在跑--class--")


zhangsan = Person("张三", 18)
zhangsan.eat()        # ---正在吃---
zhangsan.run = types.MethodType(run, zhangsan)  # 类对象zhangsan动态添加对象方法run()
zhangsan.run()  # ---正在跑---


Person.classRun = classRun    # 类Person动态添加类方法classRun()

lisi = Person("李四", 28)
lisi.eat()        # ---正在吃---
lisi.classRun()  # ---正在跑---