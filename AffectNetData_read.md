数据集说明（仅说明常用的两项）
1. Manually_Annotated（常用）
包含了40+万张人工标注的10分类的情感图片，以zip压缩包存储，如下图1。解压后，出现若干子文件夹，如下图2所示。每个子文件夹下包括若干图片，如下图3所示。此时所有类别的图片均保存在一起，通过情感类别标签文件来区别不同感情的图片。数据集大小：420299张图片。




2. Manually_Annotated_file_lists.zip（常用）
包含两个文件：training.csv和validation.csv，代表所有样本分为了训练集和验证集。
训练集大小：414799张，验证集大小：5500张；

内容组织如下：

第1列subDirectory_filePath：指示了图片保存的相对路径
第2列face_x和第3列face_y：分别指示了人脸的坐标
第4列face_width和第5列face_height：分别指示了人脸的宽和高
第6列facial_landmarks：指示了人脸特征坐标点
第7列expression：指示了情感分类，从0~10，分别表示Neutral（75374张）、Happy（134915张）、Sad（25959张）、Surprise（14590张）、Fear（6878张）、Disgust（4303张）、Anger（25382张）、Contempt（4250张）、None（无表情33588张）、Uncertain（不确定12145张）、None-Face（无人脸82915）。
实际在情感分类任务中，大多只使用前7种或前8种数据进行训练分类。所以有效样本数为：287401张或291651张。

数据集划分
train/validation和情感分类划分
1. 效果
将所有的图片按照情感类别分别划分到训练集和验证集之中。实现以下结构：



————————————————
版权声明：本文为CSDN博主「SneakateRter」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/SneakateRter/article/details/154121745