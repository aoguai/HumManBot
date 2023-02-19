#!/usr/bin/python
# -*- coding: UTF-8 -*-

from .weather import *


#####################
#       WebQA       #
#####################
def search(message):
    result = ''
    '''任务型对话，功能函数建议放在search.py中'''

    '''引用其他文件示范：天气'''
    if message.find('天气') != -1:
        print(message.replace("天气", ""))
        result = get_weather(message.replace("天气", ""))
        return result

    return result
