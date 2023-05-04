# HumManBot
兼容 GPT2、Bloom 模型、人工智能标记语言 (AIML) 和任务型对话系统 (Task) 的深度中文智能对话机器人框架

A deep Chinese intelligent dialogue robot framework that is compatible with GPT2, Bloom model, Artificial Intelligence Markup Language (AIML), and Task-oriented Dialogue System.

## 源码与介绍
本人开发运行环境：

python==3.8 torch==1.13.1+cu117 transformers==4.26.1

### 实现功能
+ 知识库匹配（AIML）回答问题
+ 任务型对话系统(Task)
+ **利用 GPT2、Bloom 等模型 完成的闲聊系统回答问题**

### 特点
+ **支持载入 GPT2、Bloom 等模型进行预测回答**
+ AIML功能采用[py3Aiml_Chinese](https://github.com/yaleimeng/py3Aiml_Chinese),可正确解析带中文pattern和模板的aiml文件
+ AIML知识库更多（共35个）
+ [敏感词库](https://github.com/observerss/textfilter)更广（共1.5W个），同时敏感词判断更合理

### 使用方法
首先下载解压你会得到一个这样一个目录结构

#### 目录结构
+ **chatbot.py** ：HumManBot的启动函数
+ **deeplearning文件夹** ：存放的是实现调用 **GPT2、Bloom** 等模型的主要代码
+ **crawler文件夹** ：存放的是任务型对话系统(Task)主要代码
+ **tool文件夹** ：存放的是用于**敏感词过滤**使用的**filter.py**文件
+ **py3Aiml_Chinese 文件夹** ： [py3Aiml_Chinese](https://github.com/yaleimeng/py3Aiml_Chinese)相关文件
+ **config.cfg** ：各种路径的配置文件
+ **resources文件夹** ：建议存放AIML知识库和敏感词库
  
### 程序处理流程
chatbot.py：
1. 预处理<br/>
    限制字数<br/>
     过滤敏感词（恶心、政治、色情、违法......）<br/>
    >当你需要增加敏感词时候直接在sensitive.txt中添加即可<br/>
    >当你需要更改敏感词库位置时候请在config.cfg中修改filter_file<br/>
2. 知识库匹配（AIML）<br/>
    基本功能：打招呼、闲聊......<br/>
    异常处理：问题太长、空白问题、找不到回复......<br/>
    情绪回答：表情、夸奖、嘲笑......<br/>
   如果匹配不到回答，进行步骤三
3. 任务型对话匹配<br/>
    如果可以**请不要完全依赖于本人提供的任务功能**，请自行修改接口和对应的匹配关键词。因为本人提供的接口随时可能失效，无法保证效果。<br/>
    你可以自行添加功能和匹配关键词。包括但不限制于**天气查询、汉字查询、空气质量查询、百科**等机器人功能<br/>
    >任务型对话系统(Task)功能函数建议放在**search.py**中<br/>
    >关键词逻辑判断建议写在**crawl.py**文件中<br/>
    >同时你可要单独写一个功能文件调用，例如给出的例子**weather.py**用于取天气<br/>
4. GPT2、Bloom 模型 进行答案预测<br/>
    如果步骤三匹配不到回答，进行 GPT2、Bloom 模型 生成闲聊回答答案

    模型相关教程请移步 [WIKI](https://github.com/aoguai/HumManBot/wiki)
   
**效果展示：**

![效果1](https://github.com/aoguai/chatbot_aiml_task_demo/blob/main/images/1.png)
![效果2](https://github.com/aoguai/chatbot_aiml_task_demo/blob/main/images/2.png)

## 更多使用教程
[WIKI](https://github.com/aoguai/HumManBot/wiki)

## 参考

demo 大体框架参考于：

[基于人工智能标记语言 (AIML)和开放域问答(WebQA)的深度智能对话模型](https://github.com/zhangbincheng1997/chatbot-aiml-webqa)

知识库匹配（AIML）部分功能使用：

[py3Aiml_Chinese](https://github.com/yaleimeng/py3Aiml_Chinese)

其他参考：

[用于中文闲聊的GPT2模型(实现了DialoGPT的MMI思想)](https://github.com/yangjianxin1/GPT2-chitchat)

[敏感词过滤的几种实现+某1w词敏感词库](https://github.com/observerss/textfilter)
