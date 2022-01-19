# Luke_paddle

## 1 简介 
本项目基于paddlepaddle框架复现了Luke预训练模型，主要复现Open Entity和SQuAD1.1数据集的结果。

**项目参考：**
- [https://github.com/studio-ousia/luke](https://github.com/studio-ousia/luke)

## 2 复现精度
>#### 在Open Entity数据集的测试效果如下表。
>复现的代码未达到论文精度，但运行原论文代码也未达到论文精度(所有超参与原论文代码一致)

|网络 |opt|batch_size|数据集|F1|F1(原论文代码)|
| :---: | :---: | :---: | :---: | :---: | :---: |
|Luke-large|AdamW|2|Open Entity|77.50|77.50|

>原论文代码运行Loss曲线及训练日志：
[原论文代码Loss曲线](pytorch_luke.png)
[原论文代码训练日志](luke_pytorch_train.log)
>
>复现代码运行Loss曲线及训练日志：
[复现代码Loss曲线](paddle_luke.png)
[复现代码训练日志](open_entity_train.log)
>
>#### 在SQuAD1.1数据集的测试效果如下表。
>由于SQuAD1.1数据集比较特殊，不提供测试集，因此对比验证集的结果
>
>在SQuAD1.1数据集上，成功复现了论文精度

|网络 |opt|batch_size|数据集|F1|F1(原论文)|EM|EM(原论文)
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
|Luke-large|AdamW|8|SQuAD1.1|94.95|95.0|89.76|89.8

>复现代码及训练日志：
[复现代码训练日志](squad_train.log)
>
## 3 运行指南
首先下载预训练权重，下载地址: 
[百度网盘](https://aistudio.baidu.com/aistudio/datasetdetail/123707)
解压至`./reading_comprehension`和`./open_entity`两个路径下

下载Open Entity数据集
[下载地址](https://cloud.tsinghua.edu.cn/f/6ec98dbd931b4da9a7f0/)
把下载好的文件解压,并把解压后的Open Entity目录下的`train.json`、`test.json`和`dev.json`复制至`./open_entity/data`，或者可以直接使用`./open_entity/data`路径下的open entity数据集

下载SQuAD1.1数据集
[下载地址](https://data.deepai.org/squad1.1.zip)
，下载解压至`./reading_comprehension/squad_data/squad`下，同时需要下载由官方提供的维基百科数据集
[下载地址](https://drive.google.com/file/d/129tDJ3ev6IdbJiKOmO6GTgNANunhO_vt/view)
, 下载解压至`./reading_comprehension/squad_data`

**代码结构**
```
├─open_entity
| ├─paddle_luke.pt           #预训练权重
| ├─data                     # 数据集文件夹
| | ├─train.json             #open entity 训练集
| | ├─dev.json               #open entity 验证集
| | ├─test.json              #open entity 测试集
| | ├─merges.txt             #tokenizer 文件
| | ├─entity_vocab.tsv       #实体词文件
| | ├─vocab.json             #tokenizer 文件
| ├─luke_model               #LUKE模型文件
| | ├─utils
| | ├─entity_vocab.py
| | ├─interwiki_db.py
| | ├─model.py   
| ├─datagenerator.py         #数据生成器文件
| ├─main.py                  #运行训练并测试
| ├─open_entity.py           #LUKE下游任务
| ├─trainer.py               #训练
| ├─utils.py
| ├─word_tokenizer.py                      
├─reading_comprehension
| ├─paddle_luke.pt           
| ├─luke_model
| | ├─utils
| | ├─model.py
| ├─squad_data
| | ├─squad
| | | ├─train-v1.1.json     # SQuAD1.1数据集训练集
| | | ├─dev-v1.1.json       # SQuAD1.1数据集验证集
| | ├─squad_change
| | ├─entity_vocab.tsv
| | ├─merges.txt
| | ├─metadata.json
| | ├─vocab.json
| | ├─enwiki_20160305.pkl   #LUKE官方提供的维基百科数据集
| | ├─enwiki_20181220_redirects.pkl   #LUKE官方提供的维基百科数据集
| | ├─enwiki_20160305_redirects.pkl   #LUKE官方提供的维基百科数据集
| ├─src
| ├─utils
| ├─create_squad_data.py
| ├─main.py
| ├─reading_comprehension.py         #LUKE下游任务                                       
```

安装第三方库
```bash
pip install -r requirements.txt
```

##### 训练并测试Luke在Open Entity数据集的精度:
###### 进入到`./open_entity`文件夹下, 运行下列命令
```bash
python main.py 2>&1 | tee train.log
```
运行结束后你将看到如下结果:
```bash
Results: %s {
  "test_f1": 0.7750939345142244,
  "test_precision": 0.7925356750823271,
  "test_recall": 0.7584033613445378
}
```

##### 训练并测试Luke在SQuAD1.1数据集的精度:
###### 进入到`./reading_comprehension`文件夹下, 运行下列命令
```bash
python create_squad_data.py
python main.py 2>&1 | tee train.log
```
运行结束后你将看到如下结果:
```bash
{"exact_match": 89.75691579943235, "f1": 94.95702001984502}
```
**说明**

1、本项目在Aistudio平台，使用Tesla V100训练  
2、本项目基于PaddlePaddle开发。  