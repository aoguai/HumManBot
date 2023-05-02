"""该文件包含PyAIML包中其他模块使用的各种通用实用函数。    """

from .LangSupport import splitChinese


def sentences(s):
    """将一堆字符串切分成一个句子列表。"""
    if not isinstance(s, str):
        raise TypeError("s must be a string")

    sentence_generator = (s[pos:end].strip() for pos, end in _get_sentence_indexes(s))

    # 自动转换中文！
    return [u' '.join(splitChinese(s)) for s in sentence_generator] or [s]


def _get_sentence_indexes(s):
    pos = 0
    l = len(s)
    while pos < l:
        p = s.find('.', pos)
        q = s.find('?', pos)
        e = s.find('!', pos)
        end = min(p if p != -1 else l,
                  q if q != -1 else l,
                  e if e != -1 else l)
        yield pos, end
        pos = end + 1
