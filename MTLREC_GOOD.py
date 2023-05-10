
import pickle
import time
from scipy import io
import numpy as np
from scipy import sparse
# from numpy import sqrt

rating_triple = io.loadmat(r'rating.mat')
# mat文件里可能有多个cell，各对应着一个dataset
# 可以用keys方法查看cell的名字, 现在要用list(mat.keys()),
# 另外，读取要用data = mat.get('名字'), 然后可以再用Numpy转为array
# print(mat.keys())
# # 可以用values方法查看各个cell的信息
# print(mat.values())
rating_triple=rating_triple['ratings']
# print(rating_triple)
# print(type(rating_triple))
rating = np.array(rating_triple)
# print(type(rating))
# print(rating)
np.random.seed(7297)

# load data
usercount = len(np.unique(rating[:, 0]))
itemcount = len(np.unique(rating[:, 1]))
#
lll = np.random.permutation(rating.shape[0])
traincount = int(np.ceil(rating.shape[0] * 0.8))
# traincount = int(np.ceil(rating.shape[0] * 0.7))
# traincount = int(np.ceil(rating.shape[0] * 0.6))

testcount = rating.shape[0] - traincount
train_rating = rating[lll[:traincount], :]
test_rating = rating[lll[traincount:], :]

# # 直接读取处理过的
# rating_train_test_ui = io.loadmat(r'origin_train_test_ciao.mat')
# train_rating=rating_train_test_ui['train_rating']
# train_rating = np.array(train_rating)
# test_rating = rating_train_test_ui['test_rating']
# test_rating = np.array(test_rating)
# UI_matrix = rating_train_test_ui['UI_train_matrix']
# UI_matrix = np.array(UI_matrix)

# np.random.seed()  # for reproducibility


# initiate parameter
k = 20
comunitycount = 20

# latent matrix
U = 0.1 * np.random.rand(usercount, k)
Q = np.random.rand(comunitycount, k)
V = 0.1 * np.random.rand(itemcount, k)
F = 0.1 * np.random.rand(usercount, k)
globalrating = np.mean(train_rating[:, 2])
# print(globalrating)
# globalrating=0;
# globalrating=0;

# bias
bu = 0.1 * np.random.rand(1, usercount)
bc = 0.1 * np.random.rand(1, comunitycount)
bj = 0.1 * np.random.rand(1, itemcount)
# print(usercount)
# print(bu[0,0])

# implicit feedback

P_rated = [np.zeros((1, k))] * usercount
Pos_rated = 0.1 * np.random.rand(itemcount, k)

# parameters of community domain
# yita = [np.zeros((1, k))] * usercount
# yitaj = [np.zeros((1, k))] * comunitycount
# yita2 = [np.zeros((1, k))] * usercount
yita = [np.zeros((1, 1))] * usercount
yitaj = [np.zeros((1, 1))] * comunitycount
yita2 = [np.zeros((1, 1))] * usercount

yita_u = 0.001 * np.random.rand(usercount, 1)
yita_j = 0.001 * np.random.rand(comunitycount, 1)

# learning rate regulazatin coefficient
eta = 0.01  # ?
eta2 = 0.01
eta3 = 0.001
eta4 = 0.01
lambda_ = 0.1  #
lambda2 = 0.1
lambda22 = 0.1
lambda3 = 2
# lambda3=2
lambda4 = 2
lambda_yita = 0.001
lambda_yita_j = 0.001
lambda_yita2 = 0.001
lambda_com = 1
lambda_f = 0
lambda_t = 0.2
shequ_zongshu = np.arange(0, comunitycount)
user_zongshu = np.arange(0, usercount)
dimk_yuanshi = np.arange(0, k)
lambda_kkk = 0.1

mapping_coficient = 1

# coefficient for different parts of predication fomula
rou = 1
rou2 = 2
rou3 = 1

P = [np.zeros((1, k))] * usercount
Pos = 0.1 * np.random.rand(comunitycount, k)


feiling_shequnei = io.loadmat(r'E:\code_project\MTLREC_MAIN\community_ciao.mat')
feiling_shequnei=feiling_shequnei['communityMatrix']
# print(feiling_shequnei)
# print(type(feiling_shequnei))
feiling_shequnei = np.array(feiling_shequnei)
# print(type(feiling_shequnei))
precommunityMatrix_juzhen = sparse.csr_matrix(feiling_shequnei)
communityMatrix = precommunityMatrix_juzhen

user_rating = io.loadmat(r'E:\code_project\MTLREC_MAIN\userrating.mat')

user_rating=user_rating['userrating']

userrating_matrix = np.array(user_rating)


UI_train_matrix = np.zeros((usercount, itemcount))
for i in range(train_rating.shape[0]):
    UI_train_matrix[train_rating[i,0]-1, train_rating[i,1]-1] = train_rating[i,2]
#
userrating = np.zeros((usercount, itemcount))

# print(UI_train_matrix[0,0])
# for i in range(usercount):
#     print(i)
#     print(np.nonzero(UI_train_matrix[i,:])[1])
#     currentindex = np.nonzero(UI_train_matrix[i,:])[1]
#     userrating[i,0]= len(currentindex)
#     userrating[i, 1:userrating[i,0]] = currentindex
# #
#
# userrating = np.zeros((usercount, 20))
#
# for i in range(usercount):
#     currentindex = np.where(UI_train_matrix[i,:] > 0)[0]
#     print(currentindex)
#     userrating[i, 0] = len(currentindex)
#     userrating[i, 1:userrating[i,0]+1] = currentindex


Com_user = [None]*comunitycount
Com_user_Count = np.random.rand(comunitycount, 1)
non_Com_user = [None]*comunitycount

for m in range(comunitycount):
    Com_user[m] = np.nonzero(communityMatrix[:, m])[0]
    Com_user_Count[m] = len(Com_user[m])
    non_Com_user[m] = np.setdiff1d(np.arange(usercount), Com_user[m])
# print(len(non_Com_user[1]))


for waixunhuan in range(1000):
    userCom = [[] for _ in range(usercount)]
    non_userCom = [[] for _ in range(usercount)]
    userComCount = np.random.randint(0, 100, (usercount, 1))
    non_userComCount = np.random.randint(0, 100, (usercount, 1))
    comComCount = np.random.randint(0, 100, (comunitycount, 1))
    non_comComCount = np.random.randint(0, 100, (comunitycount, 1))
    shequcom = [[] for _ in range(comunitycount)]
    non_shequcom = [[] for _ in range(comunitycount)]
    for m in range(usercount):
        userCom[m] = np.nonzero(precommunityMatrix_juzhen[m, :])[1]
        # print(userCom[m])
        non_userCom[m] = np.setdiff1d(shequ_zongshu, userCom[m])
        userComCount[m]= len(userCom[m])
        non_userComCount[m]= len(non_userCom[m])
    # print(userCom[2])
    # P[m] = 0.01*np.random.rand(userComCount[m],k)
    # P_rated[m]=0.01*np.random.rand(userrating[m,1],k)
    tbian = 5
    print(waixunhuan)
    if waixunhuan > 0:
        tbian = 160
        lambda_ = 0.04
        eta = 0.0005
    for ttt in range(tbian):
        tempU = np.zeros((usercount, k))
        tempV = np.zeros((itemcount, k))
        PosSum = np.power(userComCount, -0.5)  # |tru|^-0.5
        PosSum[PosSum == np.inf] = 0
        PosSum_rated = np.power(userrating_matrix[:, 0], -0.5)  # |tru|^-0.5
        PosSum_rated[PosSum_rated == np.inf] = 0
        temptk = np.zeros((comunitycount, k))

        # tempU = np.zeros((usercount, k))
        # tempV = np.zeros((itemcount, k))
        # PosSum = np.power(userComCount, -0.5)  # |tru|^-0.5
        # PosSum[np.isinf(PosSum)] = 0
        # PosSum_rated = np.power(userrating[:, 0], -0.5)  # |tru|^-0.5
        # PosSum_rated[np.isinf(PosSum_rated)] = 0
        # temptk = np.zeros((comunitycount, k))

        time_start = time.time()
        for j in range(traincount):
            currentuser = train_rating[j, 0]-1
            currentitem = train_rating[j, 1]-1

            # currentuser_trust_com = userCom[currentuser, :]  # ???¡ìu?¨´????????id

            tcount = userrating_matrix[currentuser, 0]
            # print('Iteration is ')
            # print(j)
            # print(tcount)
            neiindex = userrating_matrix[currentuser, 1:tcount]
            # print(neiindex)
            neigh = np.sum(Pos_rated[neiindex, :], axis=0)
            # print(neigh)
            ccount = userCom[currentuser]
            neigh_com = np.sum(Pos[ccount, :], axis=0)
            # print(neigh_com)

            sum_shequ = 0

            if train_rating[j, 2] != 0:
                # print(currentuser)
                # print(currentitem)
                # e = globalrating + bu[currentuser] + bj[currentitem] + sum_shequ + (rou3 * U[currentuser, :] + (
                #             rou * PosSum[currentuser, :] * neigh_com + rou2 * PosSum_rated[currentuser,
                #                                                               :] * neigh)) @ V[currentitem, :].T - \
                #     train_rating[j, 2]

                e = globalrating + bu[0,currentuser] + bj[0,currentitem] + sum_shequ + (rou3 * U[currentuser, :] + (
                            rou * PosSum[currentuser] * neigh_com + rou2 * PosSum_rated[currentuser] * neigh)) @ V[currentitem, :].T - \
                    train_rating[j, 2]

                tempU = rou3 * e * V[currentitem, :]  # ???¨²???????????????¨¹?????????¨´¡À????¨´?????¨¨??????????
                tempV = e * (rou3 * U[currentuser, :] + (
                            rou * PosSum[currentuser] * neigh_com + rou2 * PosSum_rated[currentuser] * neigh))
                temptk = e * rou * PosSum[currentuser] * V[currentitem, :]  # ??????????tk??????????tk?¨®??
                temp_rated = e * rou2 * PosSum_rated[currentuser] * V[currentitem, :]

                tempbu = e
                tempbj = e
                tempbc = e

                # if e > 1000000:
                #     return

                U[currentuser, :] = U[currentuser, :] - eta * (tempU + lambda_ * U[currentuser, :])
                V[currentitem, :] = V[currentitem, :] - eta * (tempV + lambda_ * V[currentitem, :])  # ????¡¤???¡Á??¨®????
                bu[0,currentuser] = bu[0,currentuser]  - eta * (tempbu + lambda_ * bu[0,currentuser])
                bj[0,currentitem] =bj[0,currentitem] - eta * (tempbj + lambda_ * bj[0,currentitem])
                Pos[ccount, :] = Pos[ccount, :] - eta2 * (temptk + lambda_ * Pos[ccount, :])
                Pos_rated[neiindex, :] = Pos_rated[neiindex, :] - eta2 * (temp_rated + lambda_ * Pos_rated[neiindex, :])
            # #  item ranking
            # rui_pre = globalrating + bu[0, currentuser] + bj[0, currentitem] + sum_shequ + \
            #           (rou3 * U[currentuser, :] + (
            #                       rou * PosSum[currentuser] * neigh_com + rou2 * PosSum_rated[
            #                                                                                     currentuser, :] * neigh[
            #                                                                                                       currentuser,
            #                                                                                                       :])) @ V[
            #                                                                                                              currentitem,
            #                                                                                                              :].T
            # threshold_binary = 0
            # if (train_rating(j, 2) > threshold_binary):
            #     label_y = 1
            # else:
            #     label_y = 0
            # eee1 = (1 / (1 + np.exp(-rui_pre))) - label_y
            # tempU = rou3 * eee1 * V[currentitem, :]
            # tempV = eee1 * (rou3 * U[currentuser, :] + (
            #             rou * PosSum[currentuser, :] * neigh_com[currentuser,
            #                                                      :] + rou2 * PosSum_rated[currentuser,
            #                                                                       :] * neigh[currentuser, :]))
            # temptk = eee1 * rou * PosSum[currentuser, :] * V[currentitem, :]
            # temp_rated = eee1 * rou2 * PosSum_rated[currentuser, :] * V[currentitem, :]
            # tempbu = eee1
            # tempbj = eee1
            # # if eee1 > 1000000:
            # #     return
            # U[currentuser, :] = U[currentuser, :] - eta * (
            #             tempU + lambda_ * U[currentuser, :])
            # V[currentitem, :] = V[currentitem, :] - eta * (
            #             tempV + lambda_ * V[currentitem, :])
            # bu[currentuser] = bu[currentuser] - eta * (tempbu + lambda_ * bu[currentuser])
            # bj[currentitem] = bj[currentitem] - eta * (tempbj + lambda_ * bj[currentitem])
            # Pos[ccount, :] = Pos[ccount, :] - eta2 * (temptk + lambda_ * Pos[ccount, :])
            # Pos_rated[neiindex, :] = Pos_rated[neiindex, :] - eta2 * (
            #         temp_rated + lambda_ * Pos_rated[neiindex, :])
            # Ui_PRE_test = np.zeros((usercount, itemcount))

        s = 0
        total = 0

        sqrt = np.sqrt
        for i in range(testcount):
            # for i in range(test_rating.shape[0]):
            userid = test_rating[i, 0]-1
            itemid = test_rating[i, 1]-1
            Grade = test_rating[i, 2]

            tcount = userrating_matrix[userid, 0]
            neiindex = userrating_matrix[userid, 1:tcount]
            neightt = np.sum(Pos_rated[neiindex, :], axis=0)
            ccount = userCom[userid]
            neigh_comtt = np.sum(Pos[ccount, :], axis=0)
            # print(neigh_comtt)
            sum_shequ = 0
            # print('bu')
            # print(bu[0,userid])
            # print(np.dot(U[userid, :], V[itemid, :]))
            # print( PosSum[userid] )
            # print(neigh_comtt )
            pre = globalrating + bu[0,userid] + bj[0,itemid] + sum_shequ + (rou3 * U[userid, :] + (
                            rou * PosSum[userid] * neigh_comtt + rou2 * PosSum_rated[userid] * neightt)) @ V[itemid, :].T
            # Ui_PRE_test[test_rating[i, 0] - 1, test_rating[i, 1] - 1] = pre
            s = s + abs(Grade - pre)
            total = total + (Grade - pre) ** 2
        # print(s)
        # print(testcount)
        MAE = s / testcount
        RMSE = sqrt(total / testcount)
        print('Iteration %d,the MAE of testdata is %f, the RMSE of testdata is %f\n' % (ttt, MAE, RMSE))
        time_end = time.time()
        time_c= time_end - time_start   #运行所花时间
        print('time cost', time_c, 's')


        # community matrix
        # userCom = {}
        # non_userCom = {}
        # yita = {}
        # yita2 = {}
        # userComCount = {}
        # non_userComCount = {}



        #  community
    for m in range(usercount):
            # userCom[m] = np.nonzero(precommunityMatrix_juzhen[m, :])[0]
            # non_userCom[m] = np.setdiff1d(shequ_zongshu, userCom[m])
            # userComCount[m] = len(userCom[m])
            # non_userComCount[m] = len(non_userCom[m])
            # yita[m] = 0.001 * np.random.rand(userComCount[m] * non_userComCount[m], 1)
            # yita2[m] = 0.001 * np.random.rand(userComCount[m] * non_userComCount[m], 1)
        # for m in range(usercount):
        yita[m] = 0.001 * np.random.rand(int(userComCount[m] * non_userComCount[m]), 1)
        yita2[m] = 0.001 * np.random.rand(int(userComCount[m] * non_userComCount[m]), 1)
        # print(int(userComCount[m] * non_userComCount[m]))


        # mapping preference matrix from R to A
    U1 = mapping_coficient * U.copy()

    for ttt2 in range(2):
        for dim_k in range(k):
            # print('dim k hsi',dim_k)
            quchuk = np.setdiff1d(dimk_yuanshi, dim_k)
            # print('quchuk de diyici shi ',quchuk)
            for iii in range(usercount):
                currentuser = iii
                # print('current user shi ',currentuser)
                currentuser_yita = yita[currentuser]
                currentuser_yita2 = yita2[currentuser]
                # print('currentuser_yita',currentuser_yita)
                # print('currentuser_yita2',currentuser_yita2)
                currentuser_trust_com = userCom[currentuser]
                non_currentuser_trust_com = non_userCom[currentuser]
                # print('non_currentuser_trust_com shi',non_currentuser_trust_com)
                # if (not np.isEmpty(currentuser_trust_com) and not np.isEmpty(non_currentuser_trust_com)):

                if (np.size(currentuser_trust_com) > 0 and np.size(non_currentuser_trust_com) > 0):
                    kkk = 0
                    fenzi = 0
                    fenmu = 0
                    fenzi_yita = 0
                    fenzi_yita2 = 0
                    deta_yita_j = 0
                    for shequnei in range(len(currentuser_trust_com)):
                        ZUK = communityMatrix[currentuser, currentuser_trust_com[shequnei]]
                        for lll in range(len(non_currentuser_trust_com)):
                            T_incom = 1
                            T_outncom = 1
                            # print(ZUK)
                            kkk = kkk + 1
                            # print(currentuser,dim_k,non_currentuser_trust_com[lll]-1)
                            deta_yita = U1[currentuser, dim_k] * (
                                    Q[non_currentuser_trust_com[lll], dim_k] * T_outncom - Q[
                                currentuser_trust_com[shequnei], dim_k] * T_incom).T
                            # print(shequnei)
                            # print(currentuser_trust_com.shape())
                            dangqianshequ = currentuser_trust_com[shequnei]
                            # print(type(Com_user))
                            shequlihaoyou = Com_user[dangqianshequ]
                            shequwaihaoyou = non_Com_user[dangqianshequ]
                            # print(shequlihaoyou)
                            # print(shequwaihaoyou)
                            fri = shequlihaoyou[np.random.randint(0, len(shequlihaoyou))]
                            non_fri = shequwaihaoyou[np.random.randint(0, len(shequwaihaoyou))]
                            if (U1[currentuser, dim_k] >= U1[non_fri, dim_k] and U1[currentuser, dim_k] <= U1[
                                fri, dim_k]):
                                deta_yita2 = U1[fri, dim_k] + U1[non_fri, dim_k] - 2 * U1[currentuser, dim_k]
                                currentuser_yita2[kkk - 1] = currentuser_yita2[kkk - 1] + lambda_yita2 * deta_yita2
                                currentuser_yita2[kkk - 1] = max(0, currentuser_yita2[kkk - 1])
                                fenzi_yita2 = fenzi_yita2 + currentuser_yita2[kkk - 1] * 2
                            elif (U1[currentuser, dim_k] <= U1[non_fri, dim_k] and U1[currentuser, dim_k] >= U1[
                                fri, dim_k]):
                                deta_yita2 = -1 * (U1[fri, dim_k] + U1[non_fri, dim_k] - 2 * U1[currentuser, dim_k])
                                currentuser_yita2[kkk - 1] = currentuser_yita2[kkk - 1] + lambda_yita2 * deta_yita2
                                currentuser_yita2[kkk - 1] = max(0, currentuser_yita2[kkk - 1])
                                fenzi_yita2 = fenzi_yita2 + currentuser_yita2[kkk - 1] * (-2)
                            else:
                                fenzi_yita2 = 0
                            # print('fenzi_yiya shi ',fenzi_yita2)
                            # print(kkk)
                            # print(deta_yita)
                            # if kkk>0:
                            # print('currentuser_yita is',currentuser_yita)
                            # print(kkk-1)
                            # print('currentuser_yita[0] is', len(currentuser_yita[kkk - 1]))
                            # print('currentuser_yita length is',len(currentuser_yita[kkk - 1]))

                            currentuser_yita[kkk - 1] = currentuser_yita[kkk - 1] + lambda_yita * deta_yita
                            # print(len(currentuser_yita[kkk-1]))
                            # print(kkk)
                            currentuser_yita[kkk - 1] = np.maximum(0, currentuser_yita[kkk - 1])
                            fenzi_yita += currentuser_yita[kkk - 1] * (
                                    Q[currentuser_trust_com[shequnei], dim_k] * T_incom - Q[
                                non_currentuser_trust_com[lll], dim_k] * T_outncom)
                        # print(quchuk-1)
                        fenzi += (ZUK - np.dot(U1[currentuser, quchuk],
                                               Q[currentuser_trust_com[shequnei], quchuk].T)) * Q[
                                     currentuser_trust_com[shequnei], dim_k]
                        fenmu += Q[currentuser_trust_com[shequnei], dim_k] * Q[
                            currentuser_trust_com[shequnei], dim_k]
                    yita[currentuser] = currentuser_yita
                    yita2[currentuser] = currentuser_yita2
                    YUANSHI = (fenzi + fenzi_yita + fenzi_yita2) / (fenmu + lambda3)
                    U1[currentuser, dim_k] = YUANSHI
                    if np.sum(U1[currentuser]) > 100:
                        break




        # UPDATE COMMUNITY VECTOR
    for kkkk in range(comunitycount):
        shequcom[kkkk] = np.nonzero(precommunityMatrix_juzhen[:, kkkk])[0]
        non_shequcom[kkkk] = np.setdiff1d(user_zongshu, shequcom[kkkk])
        comComCount[kkkk] = len(shequcom[kkkk])
        non_comComCount[kkkk] = len(non_shequcom[kkkk])
        yitaj[kkkk] = 0.001 * np.random.rand(int(comComCount[kkkk]) * 20, 1)


    for tt3 in range(2):
        for dim_k1 in range(20):
            quchuk2 = np.setdiff1d(dimk_yuanshi, dim_k1)
            for jjj in range(comunitycount):
                currentshequ = jjj
                currentuser_trust_com = shequcom[currentshequ]
                currentuser_yitaj = yitaj[currentshequ]
                if len(currentuser_trust_com) > 0:
                    kkk = 0
                    fenzi_q = 0
                    fenmu_q = 0
                    for shequnei in range(len(currentuser_trust_com)):
                        non_u_com = non_userCom[currentuser_trust_com[shequnei]]
                        fenzi_qq = np.zeros(k)
                        if len(non_u_com) > 0:
                            fenzi_qqq = 0
                            ZUK = communityMatrix[currentuser_trust_com[shequnei], currentshequ]

                            for lll in range(len(non_u_com)):
                                T_incom = 1
                                T_outncom = 1
                                kkk += 1
                                deta_yita = U1[currentuser_trust_com[shequnei], dim_k1] * (
                                                Q[non_u_com[lll], dim_k1] * T_outncom - Q[
                                            currentshequ, dim_k1] * T_incom)
                                currentuser_yitaj[kkk - 1] = max(0, currentuser_yitaj[
                                        kkk - 1] + lambda_yita * deta_yita)
                                fenzi_qqq = currentuser_yitaj[kkk - 1] * U1[
                                        currentuser_trust_com[shequnei], dim_k1] * T_incom
                            fenzi_q = fenzi_q + (ZUK - U1[currentuser_trust_com[shequnei], quchuk2] @ Q[
                                    currentshequ, quchuk2]) * U1[currentuser_trust_com[shequnei], dim_k1] + fenzi_qqq
                            fenmu_q = fenmu_q + U1[currentuser_trust_com[shequnei], dim_k1] * U1[
                                    currentuser_trust_com[shequnei], dim_k1]
                    yitaj[currentshequ] = currentuser_yitaj
                    YUANSHI_q = fenzi_q / (fenmu_q + lambda4)
                    Q[currentshequ, dim_k1] = YUANSHI_q

        # preference mapping A to R
    U = 4 * U1.copy()




class getuser_community:
    def __init__(self, usercount, comunitycount, trust, precommunityMatrix_juzhen, network):
        self.usercount = usercount
        self.comunitycount = comunitycount
        self.trust = trust
        self.precommunityMatrix_juzhen = precommunityMatrix_juzhen
        self.network = network
        self.shequ_u_c_geshu = np.zeros((self.usercount, self.comunitycount))

    def detect_user_community(self):
        for i in range(self.usercount):
            for user_shequ in range(self.comunitycount):
                user_suoshu_shequ = user_shequ
                current_com_user = np.nonzero(self.precommunityMatrix_juzhen[:, user_suoshu_shequ])[0]
                current_com_friend = np.intersect1d(current_com_user, self.network[i, 1:self.network[i, 1]])
                self.shequ_u_c_geshu[i, user_suoshu_shequ] = len(current_com_friend)

# main function
# if __name__ == '__main__':
#     usercount = 100
#     comunitycount = 20
#     trust = np.zeros((usercount, usercount))
#     precommunityMatrix_juzhen = np.zeros((usercount, comunitycount))
#     network = np.zeros((usercount, 22))
#
#     # initialize trust, precommunityMatrix_juzhen, network...
#
#     comm_detect = getuser_community(usercount, comunitycount, trust, precommunityMatrix_juzhen, network)
#     comm_detect.detect_community()

    # access shequ_u_c_geshu...


def Recall_1(UI_matrix_predict, userCount, testratings, N):
    res = np.zeros((1, userCount))
    cnt = 0
    for k in range(userCount):
        sort_predict_item = np.argsort(UI_matrix_predict[k, :])[::-1]
        if np.sum(testratings[:, 0] == k) == 0:
            res[0, k] = 0
            cnt += 1
            continue
        else:
            test_item = testratings[testratings[:, 0] == k, 1].T
            testlength = len(test_item[0])

        Same_item = len(np.intersect1d(sort_predict_item[:N], test_item[:testlength]))
        res[0, k] = Same_item / testlength

    resfinal = np.sum(res) / (userCount - cnt)

    return resfinal


#
# if __name__ == '__main__':
#     1