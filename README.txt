spark-lda代码说明：
.
|-spark-lda
|	|-data
|	|	|-testdata			本地测试数据，用于本地代码运行正确性测试
|	|-src
|	|	|-LDATrain.java		LDA模型训练，并将原始bid_data数据的关键字信息通过LDA生成关键字特征，并将其合并到bid_data中
|	|	|-LDAPredict.java	利用已有LDA模型预测新的关键字特征，同时对新生成的bid_data进行与训练数据同样的处理，即将原始bid_data数据的关键字信息通过LDA生成关键字特征，并将其合并到bid_data中
|	|-target
|	|-pom.xml
|	|-spark-lda.iml

运行说明：
1. 打包
   在项目目录下运行mvn package将代码打成jar包，指定类和对应参数提交spark集群运行即可。
2. LDATrain运行参数
   LDATrain需要指定7个参数，分别是
	   [1]用于训练的bid_data文件路径
	   [2]bid_data中关键字处理完成并合并后得到的新的bid_data存储路径
	   [3]训练得到的LDA模型存储路径
	   [4]LDA模型训练时指定的话题数K
	   [5]LDA模型训练时的最大迭代次数maxIterNum
	   [6]LDA模型训练时的alpha
	   [7]LDA模型训练时的beta
	必须将这7个参数全部指定才能运行代码。
	生成LDA模型后，可以利用LDAPredict进行新数据的话题特征处理。
	bid_data中关键字处理完成后，就可以进行数据特征抽取及后续LR训练。
3. LDAPredict运行参数
   LDATrain需要指定7个参数，分别是
	   [1]待处理的bid_data路径
	   [2]bid_data中关键字处理完成并合并后得到的新的bid_data存储路径
	   [3]用于LDA预测的模型文件路径
	   [4]LDA模型的话题数K
	必须将这4个参数全部指定才能运行代码。
	bid_data中关键字处理完成后，就可以进行数据特征抽取及后续LR训练。