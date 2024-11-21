# webclass
技术报告：基于机器学习的URL日志分类项目
一、项目概述
本项目旨在对训练和测试数据中的URL日志进行预处理、特征工程、以及多分类任务建模，最终根据输入数据生成预测结果文件。项目主要应用了 LightGBM 模型进行多分类任务，结合多种特征处理与提取方法（如TF-IDF特征、URL解析等），实现了对日志数据的高效分析与分类。

二、文件结构与功能说明
项目包含以下主要文件，每个文件负责不同的功能模块：

1. main.py
该文件是项目的主入口，整体流程包括：

加载和处理数据（调用 process.py 中的 load_and_process_data 方法）。
特征提取（调用 url_data.py 和 tfidf.py 中的相关方法）。
数据编码（调用 label_encode 方法）。
模型训练与预测（调用 run_lgb 方法）。
最终生成预测结果文件 results.csv。
2. process.py
负责数据的加载与初步清洗，包含以下功能：

加载训练数据（从目录中读取所有CSV文件）。
加载测试数据（文件路径为输入参数）。
将训练集和测试集合并，进行空值填充，并修正列名错误（如将 lable 修正为 label）。
3. url_data.py
负责提取与URL相关的特征，主要功能包括：

解析URL中的查询参数。
提取URL路径长度及其标准差。
检测URL中可能存在的危险关键词（如 select、union 等SQL注入或XSS攻击相关的词汇）。
提取用户代理（user_agent）的关键信息。
计算请求体（body）的长度、是否为JSON格式、以及特殊字符比例。
4. tfidf.py
负责利用TF-IDF算法提取文本特征，主要功能包括：

基于字符和单词分别提取TF-IDF特征。
使用SVD（Truncated Singular Value Decomposition）对TF-IDF矩阵降维。
多次调用特征提取函数，对URL、用户代理、请求体等列提取高维特征，并降维至固定维度。
5. lib.py
包含项目中用到的各种通用库的导入和全局配置，主要包括：

数据处理与清洗库（pandas、numpy）。
文本特征提取库（TfidfVectorizer）。
模型训练库（lightgbm）。
数据编码与分割工具（LabelEncoder、StratifiedKFold）。
日志与警告过滤配置（warnings.simplefilter）。
三、项目核心流程
该项目由以下核心步骤组成：

1. 数据加载与初步处理
调用 process.py 中的 load_and_process_data 方法：

加载训练集目录下的所有CSV文件并合并为一个数据集。
对数据中的空值进行填充（填充值为 __NaN__）。
修正训练集中错误的列名。
2. URL与日志特征提取
调用 url_data.py 中的 process_urls 方法，为数据集添加以下特征：

URL解析特征：
提取URL中的查询参数数量、最大长度、长度标准差。
提取URL路径、文件类型、路径长度、路径中的斜杠数量。
危险关键词检测：
检测URL中是否包含SQL注入或XSS相关的关键词。
用户代理特征：
提取用户代理的简写和首项信息。
请求体特征：
计算请求体长度、是否为JSON格式、特殊字符比例等。
3. TF-IDF特征提取
调用 tfidf.py 中的 tfidf 方法：

对以下列数据分别提取基于字符和单词的TF-IDF特征：
URL解码后的内容（url_unquote）。
用户代理（user_agent）。
请求体（body）。
使用SVD对上述特征进行降维，降低计算复杂性。
4. 数据编码
调用主文件中的 label_encode 方法：

对分类特征（如 method、refer、ua_short 等）进行标签编码，转化为模型可接受的数值特征。
5. LightGBM模型训练与预测
调用主文件中的 run_lgb 方法：

使用分层交叉验证（StratifiedKFold）将训练集划分为10折。
使用LightGBM进行多分类任务：
设置多分类目标函数（multiclass）。
定义类别权重（class_weight 和自定义 custom_weights）以应对类别不平衡问题。
在验证集上使用早停机制（early_stopping）。
输出OOF（Out-of-Fold）预测结果与测试集预测结果。
四、模型与算法设计
1. 特征工程
项目中特征工程涉及多种方法：

URL解析：通过对URL内容进行解析，提取结构化信息（如路径、查询参数等）。
文本特征提取：使用TF-IDF算法将文本数据转化为特征向量，结合SVD降维，保留主要信息。
危险关键词检测：基于正则匹配，检测URL中可能的攻击性关键词。
2. 模型选择
本项目选择了 LightGBM，其优势包括：

高效支持多分类任务。
能够处理大规模稀疏特征。
本项目中结合了类别权重与早停机制，进一步优化了模型性能。
3. 类别不平衡处理
通过以下方式处理类别不平衡问题：

自动计算类别权重（compute_class_weight）。
手动定义权重比例（custom_weights）。
