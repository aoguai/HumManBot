# HumManBot
Deep Chinese intelligent dialogue robot based on GPT2 model of Chinese chat, artificial intelligence markup language (AIML) and task-based dialogue system (Task).

基于中文闲聊的 GPT2 模型、人工智能标记语言 (AIML)和任务型对话系统(Task)的深度中文智能对话机器人。
## 前言
**本 demo 大体框架参考于：**

**[基于人工智能标记语言 (AIML)和开放域问答(WebQA)的深度智能对话模型](https://github.com/zhangbincheng1997/chatbot-aiml-webqa)**

**项目**

由于原项目已处于无人维护且年代久远，经本人重新修改重新适配 py3 版本，同时进行了大量修改，加入了任务型插件功能与 GPT2 模型的对话实现。

---
特此开源供各位参考学习

## 源码与介绍

### 实现功能
+ 知识库匹配（AIML）回答问题
+ 任务型对话系统(Task)
+ **利用 GPT2模型 完成的闲聊系统回答问题**

### 特点
+ **支持 GPT-2 模型的闲聊回答**
+ AIML功能采用[py3Aiml_Chinese](https://github.com/yaleimeng/py3Aiml_Chinese),可正确解析带中文pattern和模板的aiml文件
+ AIML知识库更多（共35个）
+ [敏感词库](https://github.com/observerss/textfilter)更广（共1.5W个），同时敏感词判断更合理

### 使用方法
首先下载解压你会得到一个这样一个目录结构

#### 目录结构
+ **chatbot.py** ：HumManBot的启动函数
+ **deeplearning文件夹** ：存放的是调用 **GPT2 模型** 人机交互的主要代码
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
4. GPT2 模型 人机交互<br/>
    如果步骤三匹配不到回答，进行 GPT2 模型 生成闲聊回答答案

在 `core/deeplearning/model` 中提供了由 [yangjianxin1大佬](https://github.com/yangjianxin1/GPT2-chitchat) 提供的模型与词库文件

同时你可以自行在`config.cfg`中修改对应的模型与词库路径。

模型训练方法请参考 [yangjianxin1大佬的 GPT2-chitchat 项目说明](https://github.com/yangjianxin1/GPT2-chitchat) 

|模型 | 共享地址 |模型描述|
|---------|--------|--------|
|model_epoch40_50w | [百度网盘【提取码:ju6m】](https://pan.baidu.com/s/1iEu_-Avy-JTRsO4aJNiRiA) 或 [GoogleDrive](https://drive.google.com/drive/folders/1fJ6VuBp4wA1LSMpZgpe7Hgm9dbZT5bHS?usp=sharing) |使用50w多轮对话语料训练了40个epoch，loss降到2.0左右。|

#### 学习功能
利用AIML模板+shelve存储，同时修复了已知BUG
>学习功能模板为learn.aiml库，可以自定义修改
1. * 说错 *<br/>
2. * 答错 *<br/>
3.  ......
   
   
**效果展示：**

![效果1](https://github.com/aoguai/chatbot_aiml_task_demo/blob/main/images/1.png)
![效果2](https://github.com/aoguai/chatbot_aiml_task_demo/blob/main/images/2.png)

## 更多使用教程
[WIKI](https://github.com/aoguai/HumManBot/wiki)

## 参考

[基于人工智能标记语言 (AIML)和开放域问答(WebQA)的深度智能对话模型](https://github.com/zhangbincheng1997/chatbot-aiml-webqa)

[py3Aiml_Chinese](https://github.com/yaleimeng/py3Aiml_Chinese)

[用于中文闲聊的GPT2模型(实现了DialoGPT的MMI思想)](https://github.com/yangjianxin1/GPT2-chitchat)

[敏感词过滤的几种实现+某1w词敏感词库](https://github.com/observerss/textfilter)
