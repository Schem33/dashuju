import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import cv2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import datetime
import warnings
warnings.filterwarnings('ignore')
starttime = datetime.datetime.now()
X = []
Y = []

for i in range(0, 10):
	# 遍历文件夹，读取图片
	for f in os.listdir(r"C:\Users\Azio\Desktop\hw2\Sort_1000pics\\%s" % i):
		# 打开一张图片并灰度化
		Images = cv2.imread(r"C:\Users\Azio\Desktop\hw2\Sort_1000pics/%s/%s" % (i, f))
		image = cv2.resize(Images, (256, 256), interpolation=cv2.INTER_CUBIC)
		hist = cv2.calcHist([image], [0, 1], None, [256, 256],
		                    [0.0, 255.0, 0.0, 255.0])
		X.append(((hist / 255).flatten()))
		Y.append(i)
X = np.array(X)
Y = np.array(Y)
# 切分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=60)

clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
predict_target=clf.predict(x_test)

print(sum(predict_target==y_test))
predictions_labels = clf.predict(x_test)
print(confusion_matrix(y_test, predictions_labels))
print(classification_report(y_test, predictions_labels))
endtime = datetime.datetime.now()
print(endtime - starttime)

