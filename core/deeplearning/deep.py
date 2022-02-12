#!/usr/bin/python
# -*- coding: UTF-8 -*-

from .NLP import *

import httpx, re, time



# 过滤特殊字符，保留中文、英文和数字
def filter_spec_chars(strs):
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 匹配不是中文、大小写、数字的其他字符
    return cop.sub('', strs)  # 将strsing1中匹配到的字符替换成空字符


def random_z():
    t = time.time()
    time_s = str(round(t * 1000000))
    return int(time_s[::-1][1:6])

# mcenjoy机器人回复 https://mcenjoy.cn/api/v1/chat?s=
def mcenjoy_reply(strs):
    # start_time = time.time()
    url = "https://mcenjoy.cn/api/v1/chat"
    params = {
        "s": strs
    }
    try:
        res = httpx.get(url, params=params).json()
        # print("mcenjoy：",res)
        a = res['data']
        txt = filter_spec_chars(a.replace("[cqname]", "").replace("[name]", "我").replace("米娅", "猪猪侠"))
        # elapse_time = time.time() - start_time
        # print("mcenjoy耗时："+str(elapse_time))
    except Exception as e:
        print(f"mcenjoy机器人回复请求失败\r\n {e}")
        return "我不明白"
    return txt


# 茉莉机器人回复 http://i.itpk.cn/api.php
def lili_reply(strs):
    # start_time = time.time()
    url = "http://i.itpk.cn/api.php"
    params = {
        "question": strs,
        "limit": "8",
        "api_key": "",
        "api_secret": "",
    }
    try:
        res = httpx.get(url, params=params).text
        # print("茉莉：",res)
        txt = filter_spec_chars(res.replace("[cqname]", "").replace("[name]", "我"))
        # elapse_time = time.time() - start_time
        # print("mcenjoy耗时："+str(elapse_time))
    except Exception as e:
        print(f"茉莉机器人回复请求失败\r\n {e}")
        return "我不明白"
    return txt


# 青云客回复 www.qingyunke.com
def qingyunke_reply(strs):
    # start_time = time.time()
    url = "http://api.qingyunke.com/api.php"
    params = {
        "key": "free",
        "appid": "0",
        "msg": strs
    }
    try:
        res = httpx.get(url, params=params).json()
        # print("青云客：",res)
        a = res['content']
        txt = filter_spec_chars(
            a.replace("{br}", "").replace("{face:51}", "").replace("{face:6}", "").replace("{face:3}", "").replace(
                "{face:71}", ""))
        # elapse_time = time.time() - start_time
        # print("mcenjoy耗时："+str(elapse_time))
    except Exception as e:
        print(f"青云客机器人回复请求失败\r\n {e}")
        return "我不明白"
    return txt


# ruyi机器人回复 https://ruyi.ai/index.html
def ruyi_reply(user_id, strs):
    # start_time = time.time()
    url = "http://api.ruyi.ai/v1/message"
    params = {
        "q": strs,
        "app_key": "",
        "user_id": user_id
    }
    try:
        res = httpx.get(url, params=params).json()
        # print("ruyi：",res)
        a = res['result']['intents'][0]['result']['text']
        txt = filter_spec_chars(a)
        # elapse_time = time.time() - start_time
        # print("mcenjoy耗时："+str(elapse_time))
    except Exception as e:
        print(f"ruyi机器人回复请求失败\r\n {e}")
        return "我不明白"
    return txt


#思知机器人回复 https://api.ownthink.com/
def ownthink_reply(user_id, strs):
    # start_time = time.time()
    url = "https://api.ownthink.com/bot"
    params = {
        "spoken": strs,
        "appid": "",
        "userid": str(user_id)
    }
    try:
        res = httpx.get(url, params=params).json()
        #print("思知：",res)
        a = res['data']['info']['text']
        txt = filter_spec_chars(a.replace("小思", "猪猪侠"))
        # elapse_time = time.time() - start_time
        # print("mcenjoy耗时："+str(elapse_time))
    except Exception as e:
        print(f"思知机器人回复请求失败\r\n {e}")
        return "我不明白"
    return txt


# 计算文本相似度，生成最优回复
def bot_reply(strs,FromUserId=0000000000):
    if filter_spec_chars(strs) != "":
        # print("strs:"..strs)
        txt = ""
        # 分别生成四个接口回复
        lili_txt = lili_reply(strs)
        qingyunke_txt = qingyunke_reply(strs)
        ruyi_txt = ruyi_reply(FromUserId, strs)
        ownthink_txt = ownthink_reply(FromUserId, strs)
        mcenjoy_txt = mcenjoy_reply(strs)
        # print("机器人回复#->%s", lili_txt+"丨"+qingyunke_txt+"丨"+ownthink_txt+"丨"+ruyi_txt+"丨"+mcenjoy_txt)

        # 分别计算四个接口的文本相似度
        lili_dp = comprehensive_similar(filter_spec_chars(strs), filter_spec_chars(lili_txt))
        qingyunke_dp = comprehensive_similar(filter_spec_chars(strs), filter_spec_chars(qingyunke_txt))
        ruyi_dp = comprehensive_similar(filter_spec_chars(strs), filter_spec_chars(ruyi_txt))
        ownthink_dp = comprehensive_similar(filter_spec_chars(strs), filter_spec_chars(ownthink_txt))
        mcenjoy_dp = comprehensive_similar(filter_spec_chars(strs), filter_spec_chars(mcenjoy_txt))

        # 取出最大文本相似度，并比较，返回
        dp_list = [lili_dp, ruyi_dp, qingyunke_dp, ownthink_dp, mcenjoy_dp]
        maxn = max(dp_list)
        # print("机器人回复dp#->%s","莉莉："..lili_dp.."丨青云客："..qingyunke_dp.."丨ownthink:"..ownthink_dp.."丨如意："..ruyi_dp.."丨mcenjoy：".. mcenjoy_dp)
        if (maxn == lili_dp):
            return lili_txt
        elif (maxn == qingyunke_dp):
            return qingyunke_txt
        elif (maxn == ownthink_dp):
            return ownthink_txt
        elif (maxn == ruyi_dp):
            return ruyi_txt
        elif (maxn == mcenjoy_dp):
            return mcenjoy_txt
        else:
            return lili_txt
        return "我不是很明白"
    return "我不是很明白"
