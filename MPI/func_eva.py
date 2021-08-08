#coding=utf-8

import os
import sys
import transplant





               

if __name__=='__main__':
    matlab=transplant.Matlab(jvm=False, desktop=False)
    print("开始评估，请等待！")
    matlab.main_function()

