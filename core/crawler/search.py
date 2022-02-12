#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 任务型对话功能函数文件

import httpx


# 网易新闻
def news():
    url = "https://api.apiopen.top/getWangYiNews"
    try:
        soup_html = httpx.get(url).json()
    except Exception as e:
        return ""
    news = soup_html['result']
    content = ''
    for new in news:
        title = new['title']
        content += title + '\n'
    return content
