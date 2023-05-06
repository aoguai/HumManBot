from typing import Optional, List

from .BaseLMModel import BERTLMHeadModelWrapper, AutoForCausalLMWrapper


class HumManBot:
    """
    HumManBot类用于生成聊天回复，可以基于GPT2或Bloom等模型进行聊天回复的生成
    """

    def __init__(self, tokenizer_type: str, model_path: str, tokenizer_path: str, device: str,
        max_len: int, max_history_len: int, top_k: int, top_p: float,
        temperature: float, repetition_penalty: float):
        """
        初始化函数
        :param tokenizer_type: 分词器类型，可以是'bert'或'auto'
        :type tokenizer_type: str
        :param model_path: 模型路径
        :type model_path: str
        :param tokenizer_path: 分词器路径
        :type tokenizer_path: str
        :param device: 模型运行设备，可以是'cpu'或'cuda'
        :type device: str
        :param max_len: 生成回复的最大长度
        :type max_len: int
        :param max_history_len: 对话历史的最大长度
        :type max_history_len: int
        :param top_k: Top-k采样中的k值
        :type top_k: int
        :param top_p: Top-p采样中的p值
        :type top_p: float
        :param temperature: softmax温度
        :type temperature: float
        :param repetition_penalty: 生成回复时的重复惩罚
        :type repetition_penalty: float
        """

        self.tokenizer_type = tokenizer_type
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.max_len = max_len
        self.max_history_len = max_history_len
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        if tokenizer_type == 'bert':
            self.model = BERTLMHeadModelWrapper(model_path=self.model_path, tokenizer_path=self.tokenizer_path,
                                                device=self.device, max_len=self.max_len,
                                                max_history_len=self.max_history_len, top_k=self.top_k,
                                                top_p=self.top_p, temperature=self.temperature,
                                                repetition_penalty=self.repetition_penalty)
        elif tokenizer_type == 'auto':
            self.model = AutoForCausalLMWrapper(model_path=self.model_path, tokenizer_path=self.tokenizer_path,
                                                 device=self.device, max_len=self.max_len,
                                                 max_history_len=self.max_history_len, top_k=self.top_k,
                                                 top_p=self.top_p, temperature=self.temperature,
                                                 repetition_penalty=self.repetition_penalty)
        else:
            raise ValueError(f"Unknown model type: {tokenizer_type}")

    def generate_response(self, text: str, chat_history: Optional[List[str]] = None) -> str:
        """
        生成聊天回复
        :param text: 用户输入的聊天文本
        :type text: str
        :param chat_history: 聊天历史记录
        :type chat_history: Optional[List[str]]
        :return: 聊天回复
        :rtype: str
        """

        return self.model.generate_response(text, chat_history)
