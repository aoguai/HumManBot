from transformers import AutoModelForCausalLM, GPT2LMHeadModel, BertTokenizerFast, BloomTokenizerFast
from typing import Optional, List


class BaseLMModel:
    """
    基础语言模型类，包含初始化函数和生成响应的函数。
    """

    def __init__(self, model_path: str, tokenizer_path: str, device: str, max_len: int, max_history_len: int,
                 top_k: int, top_p: float, temperature: float, repetition_penalty: float):
        """
        初始化函数

        :param model_path: 模型路径
        :type model_path: str
        :param tokenizer_path: 分词器路径
        :type tokenizer_path: str
        :param device: 训练模型的设备
        :type device: str
        :param max_len: 生成文本的最大长度
        :type max_len: int
        :param max_history_len: 用于生成响应的历史记录的最大长度
        :type max_history_len: int
        :param top_k: Top-k采样
        :type top_k: int
        :param top_p: Nucleus采样的p值
        :type top_p: float
        :param temperature: softmax温度
        :type temperature: float
        :param repetition_penalty: 重复惩罚系数
        :type repetition_penalty: float
        """

        self.max_len = max_len
        self.max_history_len = max_history_len
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.device = device
        self.history = []
        self.tokenizer = self._init_tokenizer(tokenizer_path)
        self.model = self._init_model(model_path).to(self.device)

    def _init_tokenizer(self, tokenizer_path: str):
        """
        初始化分词器

        :param tokenizer_path: 分词器路径
        :type tokenizer_path: str
        :return: 分词器
        """

        raise NotImplementedError

    def _init_model(self, model_path: str):
        """
        初始化模型

        :param model_path: 模型路径
        :type model_path: str
        :return: 模型
        """

        raise NotImplementedError

    def generate_response(self, text: str, chat_history: Optional[List[str]] = None) -> str:
        """
        生成响应

        :param text: 输入的文本
        :type text: str
        :param chat_history: 历史记录
        :type chat_history: Optional[List[str]]
        :param text: 输入的文本
        :type text: str
        :return: 生成的响应
        :rtype: str
        """
        # TODO 由于各种模型的特殊标记，如CLS、SEP等各不相同，所以想要正确的分割输入问题文本和预测答案文本很难有一个完美的解决方案，暂时先这样设计了。不兼容的可以自行修改一下。
        if chat_history:
            for utr in chat_history:
                self.history.append(self.tokenizer.encode(utr, add_special_tokens=True))
        input_ids = [self.tokenizer.cls_token_id]
        for history_id, history_utr in enumerate(self.history[-self.max_history_len:]):
            input_ids.extend(history_utr)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids = self.tokenizer(f"<s>{text}</s>", return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=self.max_len,
            do_sample=True,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty
        )
        # print(self.tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True)) # 有需要可以自行查看原始预测文本
        rets = self.tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True).strip().replace(" ", "").replace("<s>", "").replace("[CLS]", "")
        if "[SEP]" in rets:
            rets = rets.split("[SEP]")[1]
        if "</s>" in rets:
            rets = rets.split("</s>")[1]
        return rets


class GPT2LMHeadModelWrapper(BaseLMModel):
    """
    GPT2语言模型类，继承自基础语言模型类。
    """

    def _init_tokenizer(self, tokenizer_path: str):
        """
        初始化Bert分词器

        :param tokenizer_path: 分词器路径
        :type tokenizer_path: str
        :return: 分词器
        """

        return BertTokenizerFast.from_pretrained(tokenizer_path, padding_side='left')

    def _init_model(self, model_path: str):
        """
        初始化模型

        :param model_path: 模型路径
        :type model_path: str
        :return: 模型
        """

        return GPT2LMHeadModel.from_pretrained(model_path)


class BloomForCausalLMWrapper(BaseLMModel):
    """
    Bloom语言模型类，继承自基础语言模型类。
    """

    def _init_tokenizer(self, tokenizer_path: str):
        """
        初始化Bert分词器

        :param tokenizer_path: 分词器路径
        :type tokenizer_path: str
        :return: 分词器
        """
        return BloomTokenizerFast.from_pretrained(tokenizer_path)

    def _init_model(self, model_path: str):
        """
        初始化模型

        :param model_path: 模型路径
        :type model_path: str
        :return: 模型
        """

        return AutoModelForCausalLM.from_pretrained(model_path)
