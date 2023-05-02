import configparser
import os

import jieba
from .crawler import crawl
from .deeplearning.HumManBot import HumManBot
from .tool import filter
from .py3Aiml_Chinese.Kernel import Kernel

jieba.setLogLevel(jieba.logging.INFO)  # 关闭分词日志


class ChatBot:
    """
        基于 GPT-2 和 WebQA 的智能对话模型
    """

    def __init__(self, config_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.cfg')):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.load_file = config.get('resource', 'load_file')  # AIML内核指定的文件路径
        self.sensitive_file = config.get('resource', 'sensitive_file')  # 敏感词库路径
        self.vocab_file = config.get('resource', 'vocab_file')  # 词库路径
        self.gpt2_model_path = config.get('resource', 'gpt2_model_path')  # GPT2模型路径

        # 初始化分词器
        jieba.initialize()
        jieba.load_userdict(self.sensitive_file)  # 定义分词字典

        # 初始化过滤器
        self.gfw = filter.BSFilter()
        self.gfw.parse(self.sensitive_file)

        # 初始化知识库
        self.mybot = Kernel()
        self.mybot.bootstrap(learnFiles=self.load_file, commands='load aiml b')

        # 初始化学习库
        self.template = '<aiml version="1.0" encoding="UTF-8">\n{rule}\n</aiml>'
        self.category_template = '<category><pattern>{pattern}</pattern><template>{answer}</template></category>'

        # 初始化HumManBot
        self.humbot = HumManBot(model_path=self.gpt2_model_path, vocab_path=self.vocab_file)

    def response(self, message):
        # 限制字数
        if len(message) > 60:
            return self.mybot.respond('MAX')
        elif not message:
            return self.mybot.respond('MIN')
        # 过滤敏感词
        message_list = jieba.lcut(message)
        message_new = ''.join(
            [self.gfw.filter(i, "*").decode() if self.gfw.filter(i, "*").count("*") == len(i) else i for i in
             message_list])
        if '*' in message_new:
            return self.mybot.respond('过滤')
        # 结束聊天
        if message in {'exit', 'quit'}:
            return self.mybot.respond('再见')
            #   exit(2)
        # 开始聊天
        else:
            # AIML
            result = self.mybot.respond(''.join(message_list))
            # 匹配模式
            if not result.startswith('#'):
                return result
            # 搜索模式
            elif '#NONE#' in result:
                #########
                # WebQA #
                #########
                ans = crawl.search(message)
                if ans:
                    return ans
                else:
                    # Deeplearing
                    ans = self.humbot.generate_response(message)
                    return ans
            else:
                return self.mybot.respond('无答案')
