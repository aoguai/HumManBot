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

    def __init__(self, model_type="gpt2",
                 config_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.cfg')):
        config = configparser.ConfigParser()
        config.read(config_file)
        """
        初始化 ChatBot 类
        :param model_type: 模型类型，gpt2 或 bloom
        :type model_type: str
        :param config_file: 配置文件路径，默认为当前目录下的 config.cfg
        :type config_file: str
        """

        self.load_file = config.get('Resource', 'load_file')  # AIML内核指定的文件路径
        self.sensitive_file = config.get('Resource', 'sensitive_file')  # 敏感词库路径
        self.gpt2_tokenizer_path = config.get('Resource', 'gpt2_tokenizer_path')  # GPT2 tokenizer路径
        self.gpt2_model_path = config.get('Resource', 'gpt2_model_path')  # GPT2模型路径
        self.bloom_tokenizer_path = config.get('Resource', 'bloom_tokenizer_path')  # bloom tokenizer路径
        self.bloom_model_path = config.get('Resource', 'bloom_model_path')  # bloom模型路径

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

        # 初始化HumManBot
        if model_type == 'gpt2':
            self.humbot = HumManBot(model_type=model_type, model_path=self.gpt2_model_path,
                                    tokenizer_path=self.gpt2_tokenizer_path if len(
                                        self.gpt2_tokenizer_path) > 0 else self.gpt2_model_path, device=self.device,
                                    max_len=self.max_len, max_history_len=self.max_history_len, top_k=self.top_k,
                                    top_p=self.top_p,
                                    temperature=self.temperature, repetition_penalty=self.repetition_penalty)
        elif model_type == 'bloom':
            self.humbot = HumManBot(model_type=model_type, model_path=self.bloom_model_path,
                                    tokenizer_path=self.bloom_tokenizer_path if len(
                                        self.bloom_tokenizer_path) > 0 else self.bloom_model_path, device=self.device,
                                    max_len=self.max_len, max_history_len=self.max_history_len, top_k=self.top_k,
                                    top_p=self.top_p,
                                    temperature=self.temperature, repetition_penalty=self.repetition_penalty)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

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
                # WebQA
                ans = crawl.search(message)
                if ans:
                    return ans
                else:
                    # Deeplearing
                    ans = self.humbot.generate_response(message)
                    return ans
            else:
                return self.mybot.respond('无答案')
