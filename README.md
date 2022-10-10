# DeepCTC
CCL2022-CLTC赛道二解决方案

## 代码复现
* 环境准备：pip install -r requirements.txt
* 一键复现：cd ensemble_gector/command && sh ensemble.sh(结果文件保存在 ensemble_gector/logs/ensemble/cged.pred.txt)
* 逐步复现：
** 在https://www.aliyundrive.com/s/veYAbwwnCp5 下载权重，将model_single解压在common_gector文件夹下,将model_ensemble解压在ensemble_gector文件夹下
** 生成pipeline主模型结果 cd common_gector && sh ccl_2022_predict.sh 
** 生成多个集成模型结果 cd ensemble_gector/command && sh inference_track2.sh
** 生成最终集成结果 cd ensemble_gector/command && sh ensemble.sh(结果文件保存在 ensemble_gector/logs/ensemble/cged.pred.txt)
## 模型介绍
我们提出了一种基于Seq2Edit模型的pipeline语法纠错方法。首先，针对赘余、 遗漏、误用、错序错误，我们分别构造了错字、多字少字、乱序的单个纠错模型。其次，我们利 用gector模型，采用不同的预训练方法训练多个统一模型。最后，我们采用多种集成方式对各个模 型进行了集成和融合。
![image](https://github.com/wang-benqiang/DeepCTC/blob/master/images/pipeline.png)

### 单个模型
针对赘余、遗漏、误用、错序错误，我们分别构造了错字、乱序、多字少字的单个纠错模型。
具体模型如下:
#### 错字
针对错字错误，我们采用realise文本模型进行纠错，模型结构如下:
![image](https://github.com/wang-benqiang/DeepCTC/blob/master/images/s.png)

#### 乱序
针对乱序错误，我们采用BERT+CRF模型进行纠错,通过标注KEEP、LEFT、RIGHT标签来 调整字词顺序。具体模型如下:
![image](https://github.com/wang-benqiang/DeepCTC/blob/master/images/w.png)

#### 多字少字
针对多字少字错误，我们采用gector模型进行纠错，具体模型如下:
![image](https://github.com/wang-benqiang/DeepCTC/blob/master/images/r-m.png)

### 统一模型
由于单个模型的纠错结果具有片面性，因此，在单个模型纠错后，我们采用gector模型将各个任务进行统一，融合了错字、乱序、多字少字各个任务，具体模型和多字少字模型相同。

### 集成
为了提高模型的准确性和泛化性，我们训练了多个模型进行融合，具体步骤如下:
![image](https://github.com/wang-benqiang/DeepCTC/blob/master/images/ensemble.jpg)

### 其它trick

* 增加判断句子正误的二分类模型，用于数据后处理，缓解误报 
* label smooth:缓解错误标签现象
* EMA:参数层面的集成
* 差分学习率:BERT层学习率2e-5;其他层学习率2e-3 
* 参数初始化:模型其他模块与BERT采用相同的初始化方式 
* 混合精度训练:提高训练速度
* 预训练模型采用macbert、macbertcsc、macbert-large 
* 将繁体中文预先转为了简体 
* 对非中文字符、”它、她、他”字符不做处理
