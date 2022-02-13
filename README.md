# HumManBot
Deep intelligent dialogue robot based on artificial intelligence markup language (AIML) and task-based dialogue system (task)

基于人工智能标记语言 (AIML)和任务型对话系统(Task)的深度智能对话机器人demo
## 声明
**本demo基于[基于人工智能标记语言 (AIML)和开放域问答(WebQA)的深度智能对话模型](https://github.com/zhangbincheng1997/chatbot-aiml-webqa)而来**

---
特此开源供各位参考学习

## 源码与介绍

### 实现功能
+ 知识库匹配（AIML）回答问题
+ 任务型对话系统(Task)
+ 利用API完成的闲聊系统回答问题
+ **待更新....**

### 特点
+ AIML知识库更多（共35个）
+ [敏感词库](https://github.com/observerss/textfilter)更广（共1.5W个），同时敏感词判断更合理
+ 新增NLP功能，用于处理文本相似度。
  + 针对文本相似度使用了：[音形码算法](https://github.com/wenyangchou/SimilarCharactor)+余弦相似度算法
+ 更多的API接口demo可以调用（支持莉莉、青云客、ownthink、如意、mcenjoy）

### 使用方法
首先下载解压你会得到一个这样一个目录结构

#### 目录结构
  >chatbot_aiml_task_demo<br>
├─ __init__.py<br/>
└─ core<br/>
       ├─ __init__.py<br/>
       ├─ chatbot.py<br/>
       ├─ config.cfg<br/>
       ├─ crawler<br/>
       │    ├─ __init__.py<br/>
       │    ├─ crawl.py<br/>
       │    ├─ search.py<br/>
       │    ├─ stationID.json<br/>
       │    └─ weather.py<br/>
       ├─ deeplearning<br/>
       │    ├─ NLP.py<br/>
       │    ├─  .......<br/>
       │    ├─ SimilarCharactor<br/>
       │    ├─ __init__.py<br/>
       │    └─ deep.py<br/>
       ├─ log<br/>
       │    ├─ .gitkeep<br/>
       │    └─ .......<br/>
       ├─ resources<br/>
       │    ├─ Book.aiml<br/>
       │    ├─ .......<br/>
       │    ├─ save.aiml<br/>
       │    ├─ load.aiml<br/>
       │    ├─ main.aiml<br/>
       │    ├─ sensitive.txt<br/>
       │    ├─ .......<br/>
       │    └─  .......<br/>
       └─ tool<br/>
              ├─ __init__.py<br/>
              └─ filter.py<br/>
  
 其中：
 
  **chatbot.py** 是主程序
  
  **config.cfg** 是各种路径的配置文件
  
 **tool文件夹** 下是存放的是用于**敏感词过滤**使用的**filter.py**文件
 
 **resources文件夹** 建议存放AIML知识库和敏感词库
 
 **deeplearning文件夹** 存放的是NLP主要代码
 
  **crawler文件夹** 存放的是任务型对话系统(Task)主要代码
  
### 程序处理流程
1. 预处理<br/>
    限制字数<br/>
     过滤敏感词（恶心、政治、色情、违法......）<br/>
    >当你需要增加敏感词时候直接在sensitive.txt中添加即可<br/>
    >当你需要更改敏感词库位置时候请在config.cfg中修改filter_file<br/>

2.  知识库匹配（AIML）<br/>
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

4. 神经网络<br/>
    本人暂时实现不了神经网络部分功能，因此使用了API代替，同时加入了NLP文本相似度计算功能。<br/>
    >如果你需要使用API请到deep.py对应的函数中填入自己的api_key<br/>
    >在deep.py的bot_reply()函数中，你可以选择使用的API接口和个数，同时也支持你自己添加API<br/>

#### 学习功能
利用AIML模板+shelve存储，同时修复了已知BUG
>学习功能模板为learn.aiml库，可以自定义修改
1. * 说错 *<br/>
2. * 答错 *<br/>
3.  ......
   
   
**效果展示：**

![效果1](https://github.com/aoguai/chatbot_aiml_task_demo/blob/main/images/1.png)
![效果2](https://github.com/aoguai/chatbot_aiml_task_demo/blob/main/images/2.png)

## 使用教程
[WIKI](https://github.com/aoguai/HumManBot/wiki)

## 参考

[基于人工智能标记语言 (AIML)和开放域问答(WebQA)的深度智能对话模型](https://github.com/zhangbincheng1997/chatbot-aiml-webqa)

[中文相似度匹配算法](https://blog.csdn.net/chndata/article/details/41114771)

[wenyangchou/SimilarCharactor](https://github.com/wenyangchou/SimilarCharactor)

[敏感词过滤的几种实现+某1w词敏感词库](https://github.com/observerss/textfilter)
