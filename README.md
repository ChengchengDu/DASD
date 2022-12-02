# DASD
the code for paper "Domain Adaption and Summary Distillation for Unsupervised Query Focused Summarization"
## 1. 自动化数据构建流程
使用CNN/DailyMail数据进行改造，得到QFS数据集 query-CNNDM。
query-CNNDM数据集获取：链接: https://pan.baidu.com/s/1tkTHOQliFHSQbV_CJiDA6Q?pwd=qcnn 提取码: qcnn

具体流程：
1）针对目标摘要的每一句话，使用斯坦福openie抽取工具抽取实体和三元组。
2）针对每一条的openie data进行过滤和去重。 

 - 应用了词形标注工具，和人为指定了一些代词进行过滤。 
 - 如果 三元组span不是以名词开头的话，直接舍弃。 
 - 如果subject和object是代词的话直接舍弃 
 - 如果三元组包含的token数少于6或者大于15个词的话 直接舍弃 
 - 对所有的subject或者object进行去重，如果使用过就不在添加
 - 对所有获取到的三元组进行了去重，如果两个三元组rouge1>0.5舍弃。 即当前三元组会和历史三元组所有三元组计算1-gram重复度，只要和其中一个的rouge1 > 0.5 则 舍弃。
依次修改对应的参数运行 
3）构建query 
 - 针对每一条三元组， 获取对应的目标摘要（原目标摘要中包含subject或者object就作为当前的目标摘要） 
 - query构造分为两类： 一种是有实体信息 一种是没有实体信息 
如果没有实体信息，则直接将其转换成一般疑问句（query对应的答案选择的是和span重复度最高的原文句子） 
如果存在实体信息，则构建特殊疑问句。 
如果subject特殊疑问词 
否则如果object是实体 则将object换成特殊疑问词
特殊疑问词包含很多，一一对应  

4）由于使用规则构建的query可能会造成语句不通顺，因此选择back translation 进行反译。

运行脚本(需要替换具体的路径)
```
cd data_utils
bash summary_squad_utils.sh
python synth_cnn.py
```

## 2.领域自适应训练
使用GSG训练目标在目标数据集上进行训练。
构造q-gsg数据
```
python gsg_utils.py
bash train_one_cnndm.sh
```


## 3.摘要蒸馏
利用教师模型生成的伪标签进行基于特定实例的平滑操作。
```
cd DASD/train_utils
bash run_cnndm_eval.sh
bash train_one_cnndm.sh
```

注：所有shell脚本的具体路径需要自行设定
## 解码
需要将run_cnn_eval.sh中具体的模型和路径替换成自己的路径。
```
cd DASD/train_utils
bash run_cnndm_eval.sh
```
