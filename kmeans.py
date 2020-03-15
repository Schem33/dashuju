from numpy import *
from sklearn import preprocessing

#coding=utf-8

def loadDataSet(data):
    data_minmax = []
    f = open('ll.txt', 'r', encoding='utf-8')
    for v in f:
        data_minmax.append([float(v.split(',')[2]), float(v.split(',')[4])])
    minmax = preprocessing.MinMaxScaler()  # 标准化处理
    data_minmax = minmax.fit_transform(data_minmax)
    return data_minmax

#计算欧式距离的函数
def countDist(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))


#随机设置k个质心
def randCent(dataSet,k):
    n = shape(dataSet)[1]
    #初始化一个k行n列的二维数组，数组初始值全部为0，然后用mat函数将其转化为矩阵
    centroids = mat(zeros([k,n]))
    for j in range(n):
        minj = min(dataSet[:,j])
        rangej = float(max(dataSet[:,j])-minj)
        centroids[:, j] = mat(minj + rangej * random.rand(k, 1))
    return centroids

#k-Means算法核心部分
def kMeans(dataSet, k, distMeas=countDist, createCent=randCent):
    #将函数distEclud和randCent作为参数传进来，可以更好的封装kMeans函数
    m = shape(dataSet)[0] 
   
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        #下面两个for循环计算每一个样本到每一个聚类中心的距离
        for i in range(m):   #遍历样本
            minDist = inf     #inf表示无穷大，-inf表示负无穷
            minIndex = -1
            for j in range(k): #遍历聚类中心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex:
                    clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
            print(centroids)
        for cent in range(k):     #重新计算聚类中心，cent从0遍历到k
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]  #nonzero函数：返回不为0元素的行列下标组成的元组
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment
    
#显示结果
def show(dataSet, k, centroids, clusterAssment):
    from matplotlib import pyplot as plt
    numSamples, dim = dataSet.shape  #dataSet.shape返回两个值，分别赋给numSamples和dim
    mark = ['^', 'x', 'o', '*', '+']  #样本集的显#示样式
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1],mark[markIndex] ,c='b',label='i')
    mark = ['^', 'x', 'o', '*', '+']  #聚类中#心的显示样式
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1],mark[i],c='r', markersize = 12 )
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.title('未归一化数据')
    plt.show()

def main():
    dataMat = mat(loadDataSet('ll.txt'))  #加载数据集。
    dataMat = mat(dataMat)
    myCentroids, clustAssing = kMeans(dataMat,3)
    print(myCentroids)
    # quantity = pd.Series(clustAssing.labels_).value_counts()
    show(dataMat, 3, myCentroids, clustAssing)

if __name__ == '__main__':
    main()
