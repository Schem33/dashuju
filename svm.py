from sklearn import svm, metrics
import numpy as np
from sklearn import model_selection
import cv2
import os
import datetime
import warnings
from sklearn.preprocessing import binarize
from sklearn.preprocessing import LabelBinarizer
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report
starttime = datetime.datetime.now()
X = []
Y = []

for i in range(0, 10):
	    # 遍历文件夹，读取图片
	for f in os.listdir(
			    r"C:\Users\Azio\Desktop\hw2\Sort_1000pics\\%s" % i):
		    # 打开一张图片并灰度化
		    Images = cv2.imread(
			    r"C:\Users\Azio\Desktop\hw2\Sort_1000pics/%s/%s" % (i, f))
		    image = cv2.resize(Images, (256, 256),
		                       interpolation=cv2.INTER_CUBIC)
		    hist = cv2.calcHist([image], [0, 1], None, [256, 256],
		                        [0.0, 255.0, 0.0, 255.0])
		    X.append(((hist / 255).flatten()))
		    Y.append(i)
X = np.array(X)
Y = np.array(Y)
x_train,x_test,y_train,y_test=model_selection.train_test_split(X,Y,test_size=0.25,random_state=60)



clf=svm.SVC(kernel='rbf',gamma=0.025,decision_function_shape='ovo',C=7)

clf.fit(x_train, y_train)
predict_target = clf.predict(x_test)
# 预测结果与真实结果的对比
print(sum(predict_target == y_test))
predictions_labels = clf.predict(x_test)
print(confusion_matrix(y_test, predictions_labels))
print(classification_report(y_test, predictions_labels))
endtime = datetime.datetime.now()
print(endtime - starttime)
# 输出准确率，召回率，F值
# print(metrics.classification_report(y_test, predict_target))
