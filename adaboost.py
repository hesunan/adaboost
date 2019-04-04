from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf 
    for i in range(n):#遍历每个属性特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps#步长
        for j in range(-1,int(numSteps)+1):#j变化-1——10
            for inequal in ['lt', 'gt']: #在lt与gt之间切换
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #计算加权错误率
                #print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    '''
                    bestStump字典进行键值对赋值
                    '''
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []          #单层决策树数组
    m = shape(dataArr)[0]     #获得dataArr行数
    D = mat(ones((m,1))/m)   #初始化权值向量
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        #print ("D:",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#alpha更新公式alpha=0.5*ln((1-e)/e)
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  
        #print ("classEst: ",classEst.T)
        '''
        若实际标记与预测标记相同，则expon=-alpha
        若实际标记与预测标记相反，则expon=alpha
        权值更新：D(t+1)=D*exp(-alpha)/sum(D)
        '''
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))                              
        D = D/D.sum()
        #aggClassEst为预测强度
        aggClassEst += alpha*classEst
        #print ("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))#sign(x) is -1 when x<0,or 1 when x>0 
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst


def adaClassify(datToClass,classifierArr):#datToClass为待分类样例
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print (aggClassEst)
    return sign(aggClassEst)


def loadDataSet(fileName):      
    numFeat = len(open(fileName).readline().split('\t')) #获取特征数 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():#遍历文件每一行
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):#遍历除最后一列的每一列
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))#最后一列为标签列
    return dataMat,labelMat


#************x轴为真阳率，y轴为真阴率****************
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #绘制光标起始位置
    ySum = 0.0 #AUC计算方法：各个小矩形高度·的累加
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#从小到大排序
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:    #tolist（）将矩阵转化为列表
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: ",ySum*xStep)
    
    
    
if __name__ == "__main__":
    datArr, labelArr = loadDataSet("horseColicTraining2.txt")
    classifierArr, aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)
    plotROC(aggClassEst.T, labelArr)
