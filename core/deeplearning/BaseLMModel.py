import torch
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

    def generate_response(self, text: str, chat_history: Optional[List[str]] = None):
        """
        生成响应

        :param text: 输入的文本
        :type text: str
        :param chat_history: 历史记录
        :type chat_history: Optional[List[str]]
        :return: 生成的响应
        :rtype: str
        """
        raise NotImplementedError


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

    def generate_response(self, text: str, chat_history: Optional[List[str]] = None) -> str:
        """
        生成响应

        :param text: 输入的文本
        :type text: str
        :param chat_history: 历史记录
        :type chat_history: Optional[List[str]]
        :return: 生成的响应
        :rtype: str
        """

        self.model.to(self.device)
        self.model.eval()
        if chat_history is not None:
            self.history = chat_history
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.history.append(text_ids)
        # Select appropriate history
        if len(self.history) > self.max_history_len:
            history = self.history[-self.max_history_len:]
        else:
            history = self.history
        input_ids = [self.tokenizer.cls_token_id]
        for history_utr in history:
            input_ids.extend(history_utr)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids.extend(text_ids)
        input_ids.append(self.tokenizer.sep_token_id)
        input_ids = torch.tensor(input_ids).long().to(self.device).unsqueeze(0)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=self.max_len,
            do_sample=True,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=self.repetition_penalty
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        num_seps = generated_text.count(self.tokenizer.sep_token)
        if num_seps < 2:
            raise ValueError(f"Generated text does not contain at least two [SEP] tokens.\n{generated_text}")
        desired_text = generated_text.split(self.tokenizer.sep_token)[2].strip().replace(" ", "")
        return desired_text


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

    def generate_response(self, text: str, chat_history: Optional[List[str]] = None) -> str:
        """
        生成响应

        :param text: 输入的文本
        :type text: str
        :param chat_history: 历史记录
        :type chat_history: Optional[List[str]]
        :return: 生成的响应
        :rtype: str
        """

        if chat_history:
            for utr in chat_history:
                self.history.append(self.tokenizer.encode(utr, add_special_tokens=False))
        input_ids = [self.tokenizer.cls_token_id]
        for history_id, history_utr in enumerate(self.history[-self.max_history_len:]):
            input_ids.extend(history_utr)
            input_ids.append(self.tokenizer.sep_token_id)
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=self.max_len,
            do_sample=True,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=self.repetition_penalty
        )
        rets = self.tokenizer.batch_decode(outputs)
        return rets[0].strip().replace(text, "").replace('</s>', "")
