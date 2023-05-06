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
    基于 GPT2 或 Bloom 等模型 和 WebQA 的智能对话模型
    """

    def __init__(self, tokenizer_type: str = "auto",
                 config_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.cfg')):
        config = configparser.ConfigParser()
        config.read(config_file)
        """
        初始化 ChatBot 类
        :param Tokenizer_type: 分词器类型，可以是'bert'或'auto'
        :type Tokenizer_type: str
        :param config_file: 配置文件路径，默认为当前目录下的 config.cfg
        :type config_file: str
        """

        self.load_file = config.get('Resource', 'load_file')  # AIML内核指定的文件路径
        self.sensitive_file = config.get('Resource', 'sensitive_file')  # 敏感词库路径
        self.tokenizer_type = tokenizer_type  # 模型类型
        if self.tokenizer_type in ["bert", "auto"]:
            self.tokenizer_path = config.get('Resource', 'tokenizer_path')  # tokenizer路径
            self.model_path = config.get('Resource', 'model_path')  # 模型路径
        else:
            raise ValueError(f"Unknown model type: {self.tokenizer_type}")

        self.max_len = int(config.get('ModelConf', 'max_len'))  # 字符串最长的长度
        self.max_history_len = int(config.get('ModelConf', 'max_history_len'))  # 记录的最大历史长度
        self.top_k = int(config.get('ModelConf', 'top_k'))  # 从前k个概率最高的词中随机选择
        self.top_p = float(config.get('ModelConf', 'top_p'))  # 从概率累计到p的词中随机选择
        self.temperature = float(config.get('ModelConf', 'temperature'))  # softmax温度，控制生成的随机性
        self.repetition_penalty = float(config.get('ModelConf', 'repetition_penalty'))  # 用于惩罚重复的惩罚因子
        self.device = config.get('ModelConf', 'device')  # 设备类型，"cpu"或"cuda"

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

    def response(self, message: str) -> str:
        """
        ChatBot 回复函数，接收用户输入信息并生成相应的回复
        :param message: 用户输入信息
        :type message: str
        :return: ChatBot 生成的回复
        :rtype: str
        """
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
        # AIML和WebQA
        result = self.mybot.respond(''.join(message_list))
        if not result.startswith('#'):  # AIML模式
            return result
        elif '#NONE#' in result:  # 搜索模式
            ans = crawl.search(message)
            if ans:
                return ans
            else:
                return self.generate_response(message)
        else:  # 无答案
            return self.mybot.respond('无答案')

    def generate_response(self, message: str) -> str:
        """
        使用HumManBot生成回复
        :param message: 用户输入信息
        :type message: str
        :return: ChatBot 生成的回复
        :rtype: str
        """
        if not hasattr(self, 'humbot'):
            # 初始化HumManBot
            self.humbot = HumManBot(tokenizer_type=self.tokenizer_type, model_path=self.model_path,
                                    tokenizer_path=self.tokenizer_path if len(
                                        self.tokenizer_path) > 0 else self.model_path,
                                    device=self.device,
                                    max_len=self.max_len, max_history_len=self.max_history_len,
                                    top_k=self.top_k,
                                    top_p=self.top_p,
                                    temperature=self.temperature,
                                    repetition_penalty=self.repetition_penalty)
        ans = self.humbot.generate_response(message)
        return ans
