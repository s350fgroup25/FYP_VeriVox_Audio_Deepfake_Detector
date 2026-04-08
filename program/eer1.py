import os
def eer(scores=[],y_test=[]):
    target_scores = []
    nontarget_scores = []

    #将两个数组读出来
    for score,lable in zip(scores,y_test):
        if lable == 1:
            target_scores.append(score)
        else:
            nontarget_scores.append(score)

    #排序,从小到大排序
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)

    # print (target_scores)

    target_size = len(target_scores)#真实的数量
    target_position = 0
    for target_position in range(target_size):
        nontarget_size = len(nontarget_scores)#伪造的数量
        nontarget_n = nontarget_size * target_position * 1.0 / target_size   #跟真实保持同样的比例
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            print ("nontarget_scores[nontarget_position] is",  nontarget_position, nontarget_scores[nontarget_position])
            print ("target_scores[target_position] is",  target_position, target_scores[target_position])
            break

    threshold = target_scores[target_position]
    print ("threshold ",  threshold)
    eer = target_position * 1.0 / target_size
    # print ("eer  ",  eer)
    return eer,threshold
    # os.system('del '+'lfcc_lightgbm.txt')
