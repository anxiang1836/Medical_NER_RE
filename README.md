# 领域知识图谱构建

> 【概述】面向医疗领域的知识图谱构建，使用的数据集为：关于糖尿病的医学研究文章，包含大量的病历病理临床分析。项目主要通过NLP算法提取文本中的领域实体和实体间关系。

## 1. 项目内容

- 通过Brat对语料进行人工标记

  > **辅助标记**：拿到一些已知的专有词典（包括：词名 + 实体类别）。通过使用AC自动机算法，预标记出语料中出现过的已知实体。
  >
  > - **A. ahocorasick库**
  >
  > ```bash
  > conda install -c https://conda.anaconda.org/conda-forge pyahocorasick
  > ```
  >
  > - **B. ahocorasick库的使用**
  >
  > ```python
  > import ahocorasick
  > # https://blog.csdn.net/u010569893/article/details/97136696
  > def make_AC(AC, word_set):
  >     for word in word_set:
  >         AC.add_word(word,word)
  >     return AC
  > key_list = ["我爱你","爱你"]
  > AC_KEY = ahocorasick.Automaton()
  > AC_KEY = make_AC(AC_KEY, set(key_list))
  > AC_KEY.make_automaton()
  > 
  > content = "我爱你，塞北的雪，爱你，我爱你！"
  > for item in AC_KEY.iter(content):
  >     word = item[1]
  >     end = item[0]
  >     start = end - len(word) + 1
  >     print(start,end,word)
  > ```

- 对命名实体进行识别与提取
  
  - biLSTM-CRF
  
  - biLSTM-LAN
  
    > [LAN的模型简介](http://www.dataguru.cn/article-15211-1.html)
    >
    > 文章主要对模型进行科普介绍，其中包含了论文原文的链接，作者是西湖大学的，任务是词性标注。
  
  - 基于Transformer的模型（TODO）
  
- 提取命名实体间的关系
  
  - BREDS（半监督学习）
  
    > 目前还在施工中（TODO）
  
- 基于neo4j的知识图谱（TODO）

## 2. 执行步骤

1. 训练生成并训练字向量，运行`00-run_train_w2v.sh`：配置w2v的参数
2. 运行`01-run_split_build_w2idx.sh`。完成数据切分训练集/验证集/测试集，并生成全数据集上的`char2idx`与`emb_matrix`。
3. 运行`02-run_train_bilstm-crf.sh`：配置有关于模型的参数

## 工程结构

```bash
.
├── preprocess # 数据预处理/封装基本数据结构
│   ├── __init__.py
│   ├── Entity.py
│   └── Data.py
└── utils      # 工具类（文件扫描读取、日志设置）
│   ├── __init__.py
│   ├── file_operator.py
│   └── logConfig.py
├── evaluate   # 评估/预测类
│   ├── __init__.py
│   ├── evaluator.py
│   └── predictor.py
├── model      # 模型相关类
│   ├── __init__.py
│   ├── base
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   └── dnn_trainer.py
│   ├── bilstm_trainer.py
│   └── w2c_trainer.py
├── noetbook   # NoteBook
│   ├── EDA.ipynb
│   ├── TODO.ipynb
│   ├── predict.ipynb
│   └── result_compare.ipynb
├── saved_model_files  # 存储持久化的数据/模型
│   ├── README.md
│   ├── char2vec.model
│   └── char2vec_prepareData.txt
├── logs       # 训练日志存储路径
│   └── README.md
├── run_split_build_w2idx_emb.py
├── run_train_bilstm.py
├── run_train_w2v.py
├── 00-run_train_w2v.sh
├── 01-run_split_build_w2idx.sh
├── 02-run_train_bilstm-crf.sh
├── README.md
└── requirements.txt
```

