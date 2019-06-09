# AdvResearch
高级科研实训


## 2019.4.17 第一次更新
- 添加图片处理python代码，实现圆形mask输出结节位置
- 调试bug，适用于mac和windows
- *下次任务：实现方形mask输出结节位置及其前后十张的图*

## 2019.4.18 第二次更新

- 新增两个mhd文件，用来检验代码
- 更改mask形状，将圆形换成方形
- *下次任务：mask大小的参数设定、代码同时生成三张一样的图bug修改*

## 2019.4.19 面谈记录

- 医学图像解读软件安装&使用介绍
- 汇报上周工作：软件下载、读.mhd图程序的理解与改进
- *下次任务：专家识别结节的流程文档整理、数据准备*

## 2019.4.21 第三次更新

- 解决了100M以上大小文件无法push的问题（直接重新加载库）注意之后若有100MB大小以上的文件，在commit时记得跳过它，这样就不会被上传了
- 尝试过的解决办法在以下几个网站，都失败了
   - [<https://blog.csdn.net/qq_33404395/article/details/80263709>]
   - [<https://www.cnblogs.com/qmmq/p/4604862.html>]
- **注意**，调试代码时只用一份.mhd文件即可
- *下次任务：方形遮罩优化*

## 2019.4.37 面谈记录
- 老师会找医生谈
- 我们细读视频中提到的论文，做一个ppt

## 2019.5.5 面谈记录

- 汇报论文研读内容小结，详见文件夹中**.pdf**文件
- *下次任务：*
   - *继续研读论文，确定具体参数*
   - *把有标记的数据可视化*
   - *继续处理代码*
   - *实现.mhd转.dcm代码功能*
   - *先用有标记的数据做出初步结果，再细化数据*

## 2019.5.6 把论文和其概述ppt&doc上传
- 论文名称：Probability of Cancer in Pulmonary Nodules Detected on First Screening CT
- 上传.mhd读图软件

## 2019.5.9 重要进展

- 使用python的数据预处理代码现在切割出来的是50x50的矩阵了，~~下一步需要优化代码结构~~，优化完成

## 2019.5.10 讨论记录

- 代码解释，理解small_img切割算法和生成的数据格式（四维）
- 讨论生成没有结节的对照集方法，在没有结节的范围内随机切割
- 安装.mhd读图软件
- *下次任务：*
  - *生成对照集数据*

## 2019.6.9 讨论记录

上传开题ppt和报告