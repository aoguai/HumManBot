#!/usr/bin/python
# -*- coding: UTF-8 -*-

import configparser
import os
import shelve

import jieba

jieba.setLogLevel(jieba.logging.INFO)  # 关闭分词日志

from .py3Aiml_Chinese.Kernel import Kernel
from .crawler import crawl
from .deeplearning import deep
from .tool import filter


class HumManBot:
    """
        基于 AIML 和 WebQA 的智能对话模型
        1. AIML 人工智能标记语言
        2. WebQA 任务型问答
        3. Deeplearning 深度学习

        usage:
        bot = ChatBot()
        print bot.response('你好')
    """

    def __init__(self, config_file='./core/config.cfg'):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.filter_file = config.get('resource', 'filter_file')  # 敏感词库路径
        self.load_file = config.get('resource', 'load_file')
        self.save_file = config.get('resource', 'save_file')
        self.shelve_file = config.get('resource', 'shelve_file')

        # 初始化分词器
        jieba.initialize()
        jieba.load_userdict(self.filter_file)  # 定义分词字典

        # 初始化过滤器
        self.gfw = filter.DFAFilter()
        self.gfw.parse(self.filter_file)

        # 初始化知识库
        self.mybot = Kernel()
        self.mybot.bootstrap(learnFiles=self.load_file, commands='load aiml b')

        # 初始化学习库
        self.template = '<aiml version="1.0" encoding="UTF-8">\n{rule}\n</aiml>'
        self.category_template = '<category><pattern>{pattern}</pattern><template>{answer}</template></category>'

    def response(self, message):
        # 限制字数
        if len(message) > 60:
            return self.mybot.respond('MAX')
        elif len(message) == 0:
            return self.mybot.respond('MIN')

        # 过滤敏感词
        message_list = list(jieba.cut(message))
        message_new = ""
        for i in message_list:
            if self.gfw.filter(i, "*").decode().count("*") == len(i):
                message_new = message_new + self.gfw.filter(i, "*").decode()
            else:
                message_new = message_new + i
        if message_new.find("*") != -1:
            return self.mybot.respond('过滤')

        # 结束聊天
        if message == 'exit' or message == 'quit':
            return self.mybot.respond('再见')
        # 开始聊天
        else:
            ########
            # AIML #
            ########
            result = self.mybot.respond(''.join(jieba.cut(message)))
            # 匹配模式
            try:
                if result[0] != '#':
                    return result
                # 搜索模式
                elif result.find('#NONE#') != -1:
                    #########
                    # WebQA #
                    #########
                    ans = crawl.search(message)
                    if ans != '':
                        return ans
                    else:
                        ###############
                        # Deeplearing #
                        ###############
                        ans = deep.bot_reply(message, 0000000000)
                        return ans
                # 学习模式
                elif result.find('#LEARN#') != -1:
                    question = result[8:]
                    answer = message
                    self.save(question, answer)
                    return self.mybot.respond('已学习')
                # MAY BE BUG
                else:
                    return self.mybot.respond('无答案')
            except:
                return self.mybot.respond('无答案')

    def save(self, question, answer):
        # 删除上次学习缓存
        if os.path.exists("resources/shelve.db.dir"):
            os.remove("resources/shelve.db.dir")
        if os.path.exists("resources/shelve.db.dat"):
            os.remove("resources/shelve.db.dat")
        if os.path.exists("resources/shelve.db.bak"):
            os.remove("resources/shelve.db.bak")
        db = shelve.open(self.shelve_file, 'c', writeback=True)
        print(question, answer)
        db[question.replace('\n', '').replace('\r', '').replace('   ', '')] = answer.replace('\n', '').replace('\r',
                                                                                                               '').replace(
            '   ', '')
        db.sync()
        rules = []
        for r in db:
            rules.append(self.category_template.format(pattern=r, answer=db[r]))
        with open(self.save_file, 'w', encoding='utf-8') as fp:
            fp.write(self.template.format(rule='\n'.join(rules)))

    def forget(self):
        import os
        os.remove(self.save_file) if os.path.exists(self.save_file) else None
        os.remove(self.shelve_file) if os.path.exists(self.shelve_file) else None
        self.mybot.bootstrap(learnFiles=self.load_file, commands='load aiml b')
