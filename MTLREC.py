
import pickle
import time
from scipy import io
import numpy as np
from scipy import sparse


"""
Community preserving social recommendation with Cyclic 
Transfer Learning
Transactions on Information Systems
XUELIAN NI and FEI XIONG*, School of Electronic and Information Engineering, Beijing Jiaotong University, China
SHIRUI PAN, School of Information and Communication Technology, Griffith University, Australia
JIA WU, School of Computing, Macquarie University, Australia
LIANG WANG, School of Computer Science, Northwestern Polytechnical University, China
HONGSHU CHEN, School of Management and Economics, Beijing Institute of Technology, China


```

"""

# feiling_shequnei = io.loadmat(r'E:\code_project\SR-HGNN-master\data\Ciao\feiling_total.mat')
# # mat文件里可能有多个cell，各对应着一个dataset
# # 可以用keys方法查看cell的名字, 现在要用list(mat.keys()),
# # 另外，读取要用data = mat.get('名字'), 然后可以再用Numpy转为array
# # print(mat.keys())
# # # 可以用values方法查看各个cell的信息
# # print(mat.values())
# feiling_shequnei=feiling_shequnei['id_feiling']
# print(feiling_shequnei)
# print(type(feiling_shequnei))
# feiling_shequnei = np.array(feiling_shequnei)
# print(type(feiling_shequnei))
# sA = sparse.csr_matrix(feiling_shequnei)
# print(sA)
# with open("data/Ciao/feiling_shequnei.pkl", 'wb') as fo:  # 将数据写入pkl文件
#     pickle.dump(sA, fo)
# output = open('data/Ciao/feiling_shequnei.pkl', 'wb')
# pickle.dump(sA, output)
# output.close()
#
#
# #


# rating_triple = io.loadmat(r'rating.mat')
# # mat文件里可能有多个cell，各对应着一个dataset
# # 可以用keys方法查看cell的名字, 现在要用list(mat.keys()),
# # 另外，读取要用data = mat.get('名字'), 然后可以再用Numpy转为array
# # print(mat.keys())
# # # 可以用values方法查看各个cell的信息
# # print(mat.values())
# rating_triple=rating_triple['ratings']
# # print(rating_triple)
# # print(type(rating_triple))
# rating = np.array(rating_triple)
# # print(type(rating))
# # print(rating)
#
#
# # load data
#
# usercount = len(np.unique(rating[:, 0]))
# itemcount = len(np.unique(rating[:, 1]))
# #
# lll = np.random.permutation(rating.shape[0])
# traincount = int(np.ceil(rating.shape[0] * 0.8))
# # traincount = int(np.ceil(rating.shape[0] * 0.7))
# # traincount = int(np.ceil(rating.shape[0] * 0.6))
#
# testcount = rating.shape[0] - traincount
# train_rating = rating[lll[:traincount], :]
# test_rating = rating[lll[traincount:], :]



np.random.seed(7297)  # for reproducibility

# feiling_shequnei = io.loadmat(r'E:\code_project\MTLREC_MAIN\community_ciao.mat')
# # mat文件里可能有多个cell，各对应着一个dataset
# # 可以用keys方法查看cell的名字, 现在要用list(mat.keys()),
# # 另外，读取要用data = mat.get('名字'), 然后可以再用Numpy转为array
# # print(mat.keys())
# # # 可以用values方法查看各个cell的信息
# # print(mat.values())
# feiling_shequnei = feiling_shequnei['communityMatrix']
# # print(feiling_shequnei)
# # print(type(feiling_shequnei))
# feiling_shequnei = np.array(feiling_shequnei)
# # print(type(feiling_shequnei))
# precommunityMatrix_juzhen = sparse.csr_matrix(feiling_shequnei)
# communityMatrix = precommunityMatrix_juzhen
#
# user_rating = io.loadmat(r'E:\code_project\MTLREC_MAIN\userrating.mat')
# # mat文件里可能有多个cell，各对应着一个dataset
# # 可以用keys方法查看cell的名字, 现在要用list(mat.keys()),
# # 另外，读取要用data = mat.get('名字'), 然后可以再用Numpy转为array
# # print(mat.keys())
# # # 可以用values方法查看各个cell的信息
# # print(mat.values())
# user_rating = user_rating['userrating']
# # print(feiling_shequnei)
# # print(type(feiling_shequnei))
# userrating_matrix = np.array(user_rating)


# print(type(feiling_shequnei))


class MTLREC():
    def __init__(self, usercount, itemcount):
        # initiate parameter
        self.k = 20
        self.comunitycount = 20
        self.usercount = usercount
        self.itemcount = itemcount
        self.globalrating = np.mean(train_rating[:, 2])


        # learning rate regulazatin coefficient
        self.eta = 0.01  # ?
        self.eta2 = 0.01
        self.eta3 = 0.001
        self.eta4 = 0.01
        self.lambda_ = 0.1  #
        self.lambda2 = 0.1
        self.lambda22 = 0.1
        self.lambda3 = 2
        # lambda3=2
        self.lambda4 = 2
        self.lambda_yita = 0.001
        self.lambda_yita_j = 0.001
        self.lambda_yita2 = 0.001
        self.lambda_com = 1
        self.lambda_f = 0
        self.lambda_t = 0.2
        self.mapping_coficient = 1

        # coefficient for different parts of predication fomula
        self.rou = 1
        self.rou2 = 2
        self.rou3 = 1
        # latent matrix
        self.shequ_zongshu = np.arange(0, self.comunitycount)
        self.user_zongshu = np.arange(0, self.usercount)
        self.dimk_yuanshi = np.arange(0, self.k)
        self.lambda_kkk = 0.1

    def setup_matrix(self):
        self.U = 0.1 * np.random.rand(self.usercount, self.k)
        self.Q = np.random.rand(self.comunitycount, self.k)
        self.V = 0.1 * np.random.rand(self.itemcount, self.k)
        self.F = 0.1 * np.random.rand(self.usercount, self.k)
        # globalrating=0;
        # globalrating=0;

        # bias
        self.bu = 0.1 * np.random.rand(1, self.usercount)
        # bc = 0.1 * np.random.rand(1, self.comunitycount)
        self.bj = 0.1 * np.random.rand(1, self.itemcount)
        # print(usercount)
        # print(bu[0,0])

        # implicit feedback
        self.P_rated = [np.zeros((1, self.k))] * self.usercount
        self.Pos_rated = 0.1 * np.random.rand(self.itemcount, self.k)

        # parameters of community domain
        self.yita = [np.zeros((1, self.k))] * self.usercount
        self.yitaj = [np.zeros((1, self.k))] * self.comunitycount
        self.yita2 = [np.zeros((1, self.k))] * usercount
        self.yita_u = 0.001 * np.random.rand(self.usercount, 1)
        self.yita_j = 0.001 * np.random.rand(self.comunitycount, 1)



        self.P = [np.zeros((1, self.k))] * usercount
        self.Pos = 0.1 * np.random.rand(self.comunitycount, self.k)

        self.Com_user = [None] * self.comunitycount
        self.Com_user_Count = np.random.rand( self.comunitycount, 1)
        self.non_Com_user = [None] * self.comunitycount

        for m in range(self.comunitycount):
            self.Com_user[m] = np.nonzero(communityMatrix[:, m])[0]
            self.Com_user_Count[m] = len(self.Com_user[m])
            self.non_Com_user[m] = np.setdiff1d(np.arange(usercount), self.Com_user[m])

        self.userCom = [[] for _ in range(self.usercount)]
        self.non_userCom = [[] for _ in range(self.usercount)]
        self.userComCount = np.random.randint(0, 100, (self.usercount, 1))
        self.non_userComCount = np.random.randint(0, 100, (self.usercount, 1))
        self.comComCount = np.random.randint(0, 100, (self.comunitycount, 1))
        self.non_comComCount = np.random.randint(0, 100, (self.comunitycount, 1))
        self.shequcom = [[] for _ in range(self.comunitycount)]
        self.non_shequcom = [[] for _ in range(self.comunitycount)]
        for m in range(usercount):
            self.userCom[m] = np.nonzero(precommunityMatrix_juzhen[m, :])[1]
            # print(userCom[m])
            self.non_userCom[m] = np.setdiff1d(self.shequ_zongshu, self.userCom[m])
            self.userComCount[m] = len(self.userCom[m])
            self.non_userComCount[m] = len(self.non_userCom[m])


    # print(len(non_Com_user[1]))
    def train(self):
        time_start = time.time()
        self.setup_matrix()
        for waixunhuan in range(1000):

            # print(userCom[2])
            # P[m] = 0.01*np.random.rand(userComCount[m],k)
            # P_rated[m]=0.01*np.random.rand(userrating[m,1],k)
            tbian = 5
            print(waixunhuan)
            if waixunhuan > 0:
                tbian = 160
                self.lambda_ = 0.05
                self.eta = 0.0005
            for ttt in range(tbian):
                # tempU = np.zeros((self.usercount, self.k))
                # tempV = np.zeros((self.itemcount, self.k))
                PosSum = np.power(self.userComCount, -0.5)  # |tru|^-0.5
                PosSum[PosSum == np.inf] = 0
                PosSum_rated = np.power(userrating_matrix[:, 0], -0.5)  # |tru|^-0.5
                PosSum_rated[PosSum_rated == np.inf] = 0
                # temptk = np.zeros((self.comunitycount, self.k))

                # tempU = np.zeros((usercount, k))
                # tempV = np.zeros((itemcount, k))
                # PosSum = np.power(userComCount, -0.5)  # |tru|^-0.5
                # PosSum[np.isinf(PosSum)] = 0
                # PosSum_rated = np.power(userrating[:, 0], -0.5)  # |tru|^-0.5
                # PosSum_rated[np.isinf(PosSum_rated)] = 0
                # temptk = np.zeros((comunitycount, k))

                for j in range(traincount):
                    currentuser = train_rating[j, 0] - 1
                    currentitem = train_rating[j, 1] - 1

                    tcount = userrating_matrix[currentuser, 0]
                    # print('Iteration is ')
                    # print(j)
                    # print(tcount)
                    neiindex = userrating_matrix[currentuser, 1:tcount]
                    neigh = np.sum(self.Pos_rated[neiindex, :], axis=0)
                    ccount = self.userCom[currentuser]
                    neigh_com = np.sum(self.Pos[ccount, :], axis=0)

                    sum_shequ = 0

                    if train_rating[j, 2] != 0:
                        # print(currentuser)
                        # print(currentitem)
                        # e = globalrating + bu[currentuser] + bj[currentitem] + sum_shequ + (rou3 * U[currentuser, :] + (
                        #             rou * PosSum[currentuser, :] * neigh_com + rou2 * PosSum_rated[currentuser,
                        #                                                               :] * neigh)) @ V[currentitem, :].T - \
                        #     train_rating[j, 2]

                        e = self.globalrating + self.bu[0, currentuser] + self.bj[0, currentitem] + sum_shequ + (
                                self.rou3 * self.U[currentuser, :] + (
                                self.rou * PosSum[currentuser] * neigh_com + self.rou2 * PosSum_rated[currentuser] * neigh)) @ self.V[
                                                                                                                     currentitem,
                                                                                                                     :].T - \
                            train_rating[j, 2]

                        tempU = self.rou3 * e * self.V[currentitem,
                                           :]  # ???¨²???????????????¨¹?????????¨´¡À????¨´?????¨¨??????????
                        tempV = e * (self.rou3 * self.U[currentuser, :] + (
                                self.rou * PosSum[currentuser] * neigh_com + self.rou2 * PosSum_rated[currentuser] * neigh))
                        temptk = e * self.rou * PosSum[currentuser] * self.V[currentitem, :]  # ??????????tk??????????tk?¨®??
                        temp_rated = e * self.rou2 * PosSum_rated[currentuser] * self.V[currentitem, :]

                        tempbu = e
                        tempbj = e

                        # if e > 1000000:
                        #     return

                        self.U[currentuser, :] = self.U[currentuser, :] - self.eta * (tempU + self.lambda_ * self.U[currentuser, :])
                        self.V[currentitem, :] = self.V[currentitem, :] - self.eta * (
                                tempV +  self.lambda_ * self.V[currentitem, :])  # ????¡¤???¡Á??¨®????
                        self.bu[0, currentuser] = self.bu[0, currentuser] -  self.eta * (tempbu +  self.lambda_ * self.bu[0, currentuser])
                        self.bj[0, currentitem] = self.bj[0, currentitem] -  self.eta * (tempbj +  self.lambda_ * self.bj[0, currentitem])
                        self.Pos[ccount, :] = self.Pos[ccount, :] - self.eta2 * (temptk +  self.lambda_ * self.Pos[ccount, :])
                        self.Pos_rated[neiindex, :] = self.Pos_rated[neiindex, :] - self.eta2 * (
                                temp_rated +  self.lambda_ * self.Pos_rated[neiindex, :])

                s = 0
                total = 0

                # Ui_PRE_test = np.zeros((usercount, itemcount))
                time_end = time.time()
                time_c = time_end - time_start  # 运行所花时间
                print('time cost', time_c, 's')

                for i in range(testcount):
                    # for i in range(test_rating.shape[0]):
                    userid = test_rating[i, 0] - 1
                    itemid = test_rating[i, 1] - 1
                    Grade = test_rating[i, 2]

                    tcount = userrating_matrix[userid, 0]
                    neiindex = userrating_matrix[userid, 1:tcount]
                    neightt = np.sum(self.Pos_rated[neiindex, :], axis=0)
                    ccount = self.userCom[userid]
                    neigh_comtt = np.sum(self.Pos[ccount, :], axis=0)
                    # print(neigh_comtt)
                    sum_shequ = 0
                    # print('bu')
                    # print(bu[0,userid])
                    # print(np.dot(U[userid, :], V[itemid, :]))
                    # print( PosSum[userid] )
                    # print(neigh_comtt )
                    pre = self.globalrating + self.bu[0, userid] + self.bj[0, itemid] + sum_shequ + (self.rou3 * self.U[userid, :] + (
                            self.rou * PosSum[userid] * neigh_comtt + self.rou2 * PosSum_rated[userid] * neightt)) @ self.V[itemid,
                                                                                                           :].T
                    # Ui_PRE_test[test_rating[i, 0] - 1, test_rating[i, 1] - 1] = pre
                    s = s + abs(Grade - pre)
                    total = total + (Grade - pre) ** 2
                # print(s)
                # print(testcount)
                MAE = s / testcount
                RMSE = np.sqrt(total / testcount)
                print('Iteration %d,the MAE of testdata is %f, the RMSE of testdata is %f\n' % (ttt, MAE, RMSE))
                time_end = time.time()
                time_c = time_end - time_start  # 运行所花时间
                print('time cost', time_c, 's')

                # community matrix
                # userCom = {}
                # non_userCom = {}
                # yita = {}
                # yita2 = {}
                # userComCount = {}
                # non_userComCount = {}

                # #  community
                # for m in range(usercount):
                #     # userCom[m] = np.nonzero(precommunityMatrix_juzhen[m, :])[0]
                #     # non_userCom[m] = np.setdiff1d(shequ_zongshu, userCom[m])
                #     # userComCount[m] = len(userCom[m])
                #     # non_userComCount[m] = len(non_userCom[m])
                #     # yita[m] = 0.001 * np.random.rand(userComCount[m] * non_userComCount[m], 1)
                #     # yita2[m] = 0.001 * np.random.rand(userComCount[m] * non_userComCount[m], 1)
                # # for m in range(usercount):
                #     yita[m] = 0.001 * np.random.rand(int(userComCount[m] * non_userComCount[m]), 1)
                #     yita2[m] = 0.001 * np.random.rand(int(userComCount[m] * non_userComCount[m]), 1)
                # # print(yita[1])
                #
                #
                # # mapping preference matrix from R to A
                # U1 = mapping_coficient * U.copy()
                #
                # # cyclic transfer learning
                # # update preference vector in community domain
                # # for ttt2 in range(1, 2):
                # #     for dim_k in range(k):
                # #         # print('dim k hsi',dim_k)
                # #         quchuk = np.setdiff1d(dimk_yuanshi, dim_k+1)
                # #         # print('quchuk de diyici shi ',quchuk)
                # #         for iii in range(usercount):
                # #             currentuser = iii
                # #             # print('current user shi ',currentuser)
                # #             currentuser_yita = yita[currentuser]
                # #             currentuser_yita2 = yita2[currentuser]
                # #             # print('currentuser_yita',currentuser_yita)
                # #             # print('currentuser_yita2',currentuser_yita2)
                # #             currentuser_trust_com = userCom[currentuser]
                # #             non_currentuser_trust_com = non_userCom[currentuser]
                # #             # print('non_currentuser_trust_com shi',non_currentuser_trust_com)
                # #             # if (not np.isEmpty(currentuser_trust_com) and not np.isEmpty(non_currentuser_trust_com)):
                # #
                # #             if (np.size(currentuser_trust_com) > 0 and np.size(non_currentuser_trust_com) > 0):
                # #                 kkk = 0
                # #                 fenzi = 0
                # #                 fenmu = 0
                # #                 fenzi_yita = 0
                # #                 fenzi_yita2 = 0
                # #                 deta_yita_j = 0
                # #                 for shequnei in range(len(currentuser_trust_com)):
                # #                     ZUK = communityMatrix[currentuser, currentuser_trust_com[shequnei]-1]
                # #                     for lll in range(len(non_currentuser_trust_com)):
                # #                         T_incom = 1
                # #                         T_outncom = 1
                # #                         # print(ZUK)
                # #                         kkk = kkk + 1
                # #                         # print(currentuser,dim_k,non_currentuser_trust_com[lll]-1)
                # #                         deta_yita = U1[currentuser, dim_k] * (
                # #                                     Q[non_currentuser_trust_com[lll]-1, dim_k] * T_outncom - Q[
                # #                                 currentuser_trust_com[shequnei]-1, dim_k] * T_incom).T
                # #                         # print(shequnei)
                # #                         # print(currentuser_trust_com.shape())
                # #                         dangqianshequ = currentuser_trust_com[shequnei]
                # #                         # print(type(Com_user))
                # #                         shequlihaoyou = Com_user[dangqianshequ]
                # #                         shequwaihaoyou = non_Com_user[dangqianshequ]
                # #                         # print(shequlihaoyou)
                # #                         # print(shequwaihaoyou)
                # #                         fri = shequlihaoyou[np.random.randint(0, len(shequlihaoyou))]
                # #                         non_fri = shequwaihaoyou[np.random.randint(0, len(shequwaihaoyou))]
                # #                         if (U1[currentuser, dim_k] >= U1[non_fri, dim_k] and U1[currentuser, dim_k] <= U1[
                # #                             fri, dim_k]):
                # #                             deta_yita2 = U1[fri, dim_k] + U1[non_fri, dim_k] - 2 * U1[currentuser, dim_k]
                # #                             currentuser_yita2[kkk-1] = currentuser_yita2[kkk-1] + lambda_yita2 * deta_yita2
                # #                             currentuser_yita2[kkk-1] = max(0, currentuser_yita2[kkk-1])
                # #                             fenzi_yita2 = fenzi_yita2 + currentuser_yita2[kkk-1] * 2
                # #                         elif (U1[currentuser, dim_k] <= U1[non_fri, dim_k] and U1[currentuser, dim_k] >= U1[
                # #                             fri, dim_k]):
                # #                             deta_yita2 = -1 * (U1[fri, dim_k] + U1[non_fri, dim_k] - 2 * U1[currentuser, dim_k])
                # #                             currentuser_yita2[kkk-1] = currentuser_yita2[kkk-1] + lambda_yita2 * deta_yita2
                # #                             currentuser_yita2[kkk-1] = max(0, currentuser_yita2[kkk-1])
                # #                             fenzi_yita2 = fenzi_yita2 + currentuser_yita2[kkk-1] * (-2)
                # #                         else:
                # #                             fenzi_yita2 = 0
                # #                         # print('fenzi_yiya shi ',fenzi_yita2)
                # #                         # print(kkk)
                # #                         # print(deta_yita)
                # #                         # if kkk>0:
                # #                             # print('currentuser_yita is',currentuser_yita)
                # #                             # print(kkk-1)
                # #                             # print('currentuser_yita[0] is', len(currentuser_yita[kkk - 1]))
                # #                             # print('currentuser_yita length is',len(currentuser_yita[kkk - 1]))
                # #
                # #                         currentuser_yita[kkk-1] = currentuser_yita[kkk-1] + lambda_yita * deta_yita
                # #                         # print(len(currentuser_yita[kkk-1]))
                # #                         # print(kkk)
                # #                         currentuser_yita[kkk-1] = np.maximum(0, currentuser_yita[kkk-1])
                # #                         fenzi_yita += currentuser_yita[kkk-1] * (
                # #                                     Q[currentuser_trust_com[shequnei]-1, dim_k] * T_incom - Q[
                # #                                 non_currentuser_trust_com[lll]-1, dim_k] * T_outncom)
                # #                     # print(quchuk-1)
                # #                     fenzi += (ZUK - np.dot(U1[currentuser, quchuk-1],
                # #                                            Q[currentuser_trust_com[shequnei]-1, quchuk-1].T)) * Q[
                # #                                  currentuser_trust_com[shequnei]-1, dim_k]
                # #                     fenmu += Q[currentuser_trust_com[shequnei]-1, dim_k] * Q[
                # #                         currentuser_trust_com[shequnei]-1, dim_k]
                # #                 yita[currentuser] = currentuser_yita
                # #                 yita2[currentuser] = currentuser_yita2
                # #                 # print('yita走了一遍')
                # #                 YUANSHI = (fenzi + fenzi_yita + fenzi_yita2) / (fenmu + lambda3)
                # #                 U1[currentuser, dim_k] = YUANSHI
                # #                 if np.sum(U1[currentuser]) > 100:
                # #                     break
                #
                #
                #
                # for ttt2 in range(1, 2):
                #     for dim_k in range(k):
                #         # print('dim k hsi',dim_k)
                #         quchuk = np.setdiff1d(dimk_yuanshi, dim_k)
                #         # print('quchuk de diyici shi ',quchuk)
                #         for iii in range(usercount):
                #             currentuser = iii
                #             # print('current user shi ',currentuser)
                #             currentuser_yita = yita[currentuser]
                #             currentuser_yita2 = yita2[currentuser]
                #             # print('currentuser_yita',currentuser_yita)
                #             # print('currentuser_yita2',currentuser_yita2)
                #             currentuser_trust_com = userCom[currentuser]
                #             non_currentuser_trust_com = non_userCom[currentuser]
                #             # print('non_currentuser_trust_com shi',non_currentuser_trust_com)
                #             # if (not np.isEmpty(currentuser_trust_com) and not np.isEmpty(non_currentuser_trust_com)):
                #
                #             if (np.size(currentuser_trust_com) > 0 and np.size(non_currentuser_trust_com) > 0):
                #                 kkk = 0
                #                 fenzi = 0
                #                 fenmu = 0
                #                 fenzi_yita = 0
                #                 fenzi_yita2 = 0
                #                 deta_yita_j = 0
                #                 for shequnei in range(len(currentuser_trust_com)):
                #                     ZUK = communityMatrix[currentuser, currentuser_trust_com[shequnei]]
                #                     for lll in range(len(non_currentuser_trust_com)):
                #                         T_incom = 1
                #                         T_outncom = 1
                #                         # print(ZUK)
                #                         kkk = kkk + 1
                #                         # print(currentuser,dim_k,non_currentuser_trust_com[lll]-1)
                #                         deta_yita = U1[currentuser, dim_k] * (
                #                                     Q[non_currentuser_trust_com[lll], dim_k] * T_outncom - Q[
                #                                 currentuser_trust_com[shequnei], dim_k] * T_incom).T
                #                         # print(shequnei)
                #                         # print(currentuser_trust_com.shape())
                #                         dangqianshequ = currentuser_trust_com[shequnei]
                #                         # print(type(Com_user))
                #                         shequlihaoyou = Com_user[dangqianshequ]
                #                         shequwaihaoyou = non_Com_user[dangqianshequ]
                #                         # print(shequlihaoyou)
                #                         # print(shequwaihaoyou)
                #                         fri = shequlihaoyou[np.random.randint(0, len(shequlihaoyou))]
                #                         non_fri = shequwaihaoyou[np.random.randint(0, len(shequwaihaoyou))]
                #                         if (U1[currentuser, dim_k] >= U1[non_fri, dim_k] and U1[currentuser, dim_k] <= U1[
                #                             fri, dim_k]):
                #                             deta_yita2 = U1[fri, dim_k] + U1[non_fri, dim_k] - 2 * U1[currentuser, dim_k]
                #                             currentuser_yita2[kkk-1] = currentuser_yita2[kkk-1] + lambda_yita2 * deta_yita2
                #                             currentuser_yita2[kkk-1] = max(0, currentuser_yita2[kkk-1])
                #                             fenzi_yita2 = fenzi_yita2 + currentuser_yita2[kkk-1] * 2
                #                         elif (U1[currentuser, dim_k] <= U1[non_fri, dim_k] and U1[currentuser, dim_k] >= U1[
                #                             fri, dim_k]):
                #                             deta_yita2 = -1 * (U1[fri, dim_k] + U1[non_fri, dim_k] - 2 * U1[currentuser, dim_k])
                #                             currentuser_yita2[kkk-1] = currentuser_yita2[kkk-1] + lambda_yita2 * deta_yita2
                #                             currentuser_yita2[kkk-1] = max(0, currentuser_yita2[kkk-1])
                #                             fenzi_yita2 = fenzi_yita2 + currentuser_yita2[kkk-1] * (-2)
                #                         else:
                #                             fenzi_yita2 = 0
                #                         # print('fenzi_yiya shi ',fenzi_yita2)
                #                         # print(kkk)
                #                         # print(deta_yita)
                #                         # if kkk>0:
                #                             # print('currentuser_yita is',currentuser_yita)
                #                             # print(kkk-1)
                #                             # print('currentuser_yita[0] is', len(currentuser_yita[kkk - 1]))
                #                             # print('currentuser_yita length is',len(currentuser_yita[kkk - 1]))
                #
                #                         currentuser_yita[kkk-1] = currentuser_yita[kkk-1] + lambda_yita * deta_yita
                #                         # print(len(currentuser_yita[kkk-1]))
                #                         # print(kkk)
                #                         currentuser_yita[kkk-1] = np.maximum(0, currentuser_yita[kkk-1])
                #                         fenzi_yita += currentuser_yita[kkk-1] * (
                #                                     Q[currentuser_trust_com[shequnei], dim_k] * T_incom - Q[
                #                                 non_currentuser_trust_com[lll], dim_k] * T_outncom)
                #                     # print(quchuk-1)
                #                     fenzi += (ZUK - np.dot(U1[currentuser, quchuk],
                #                                            Q[currentuser_trust_com[shequnei], quchuk].T)) * Q[
                #                                  currentuser_trust_com[shequnei], dim_k]
                #                     fenmu += Q[currentuser_trust_com[shequnei], dim_k] * Q[
                #                         currentuser_trust_com[shequnei], dim_k]
                #                 yita[currentuser] = currentuser_yita
                #                 yita2[currentuser] = currentuser_yita2
                #                 # print('yita走了一遍')
                #                 YUANSHI = (fenzi + fenzi_yita + fenzi_yita2) / (fenmu + lambda3)
                #                 U1[currentuser, dim_k] = YUANSHI
                #                 if np.sum(U1[currentuser]) > 100:
                #                     break
                #
                #
                #
                # # UPDATE COMMUNITY VECTOR
                # for kkkk in range(comunitycount):
                #     shequcom[kkkk] = np.nonzero(precommunityMatrix_juzhen[:, kkkk])[0]
                #     non_shequcom[kkkk] = np.setdiff1d(user_zongshu, shequcom[kkkk])
                #     comComCount[kkkk] = len(shequcom[kkkk])
                #     non_comComCount[kkkk] = len(non_shequcom[kkkk])
                #     yitaj[kkkk] = 0.001 * np.random.rand(int(comComCount[kkkk]) * 20, 1)
                #
                # for tt3 in range(1):
                #     for dim_k1 in range(20):
                #         quchuk2 = np.setdiff1d(dimk_yuanshi, dim_k1)
                #         for jjj in range(comunitycount):
                #             currentshequ = jjj
                #             currentuser_trust_com = shequcom[currentshequ]
                #             currentuser_yitaj = yitaj[currentshequ]
                #             if len(currentuser_trust_com) > 0:
                #                 kkk = 0
                #                 fenzi_q = 0
                #                 fenmu_q = 0
                #                 for shequnei in range(len(currentuser_trust_com)):
                #                     non_u_com = non_userCom[currentuser_trust_com[shequnei]]
                #                     fenzi_qq = np.zeros(k)
                #                     if len(non_u_com) > 0:
                #                         fenzi_qqq = 0
                #                         ZUK = communityMatrix[currentuser_trust_com[shequnei], currentshequ]

                #                         for lll in range(len(non_u_com)):
                #                             T_incom = 1
                #                             T_outncom = 1
                #                             kkk += 1
                #                             deta_yita = U1[currentuser_trust_com[shequnei], dim_k1] * (
                #                                         Q[non_u_com[lll], dim_k1] * T_outncom - Q[
                #                                     currentshequ, dim_k1] * T_incom)
                #                             currentuser_yitaj[kkk - 1] = max(0, currentuser_yitaj[
                #                                 kkk - 1] + lambda_yita * deta_yita)
                #                             fenzi_qqq = currentuser_yitaj[kkk - 1] * U1[
                #                                 currentuser_trust_com[shequnei], dim_k1] * T_incom
                #                         fenzi_q = fenzi_q + (ZUK - U1[currentuser_trust_com[shequnei], quchuk2] @ Q[
                #                             currentshequ, quchuk2]) * U1[currentuser_trust_com[shequnei], dim_k1] + fenzi_qqq
                #                         fenmu_q = fenmu_q + U1[currentuser_trust_com[shequnei], dim_k1] * U1[
                #                             currentuser_trust_com[shequnei], dim_k1]
                #                 yitaj[currentshequ] = currentuser_yitaj
                #                 YUANSHI_q = fenzi_q / (fenmu_q + lambda4)
                #                 Q[currentshequ, dim_k1] = YUANSHI_q
                #
                # # preference mapping A to R
                # U = 4 * U1.copy()

                # rank
                # BuMf = np.tile(bu, (itemcount, 1)).T
                # BiMf = np.tile(bj, (usercount, 1))
                # Ui_PRE = BuMf + BiMf + globalrating + rou3 * U.dot(V.T) + (
                #             ((rou * PosSum).dot(neigh_comtt) + rou2 * PosSum_rated.dot(neightt)).T * V.T).T

                # for i in range(test_rating.shape[0]):

                #
                # Ui_PRE_test[Ui_PRE_test <= 3] = 0
                # Ui_PRE_test[Ui_PRE_test > 3] = 1
                #
                # RECALL_K = Recall_1(Ui_PRE_test, usercount, test_new_rating, 10)
                # ndcg_K = NDCG(10, Ui_PRE_test, UI_test_matrix)

    # class getuser_community:
    #     def __init__(self, usercount, comunitycount, trust, precommunityMatrix_juzhen, network):
    #         self.usercount = usercount
    #         self.comunitycount = comunitycount
    #         self.trust = trust
    #         self.precommunityMatrix_juzhen = precommunityMatrix_juzhen
    #         self.network = network
    #         self.shequ_u_c_geshu = np.zeros((self.usercount, self.comunitycount))
    #
    #     def detect_user_community(self):
    #         for i in range(self.usercount):
    #             for user_shequ in range(self.comunitycount):
    #                 user_suoshu_shequ = user_shequ
    #                 current_com_user = np.nonzero(self.precommunityMatrix_juzhen[:, user_suoshu_shequ])[0]
    #                 current_com_friend = np.intersect1d(current_com_user, self.network[i, 1:self.network[i, 1]])
    #                 self.shequ_u_c_geshu[i, user_suoshu_shequ] = len(current_com_friend)

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

    # def Recall_1(UI_matrix_predict, userCount, testratings, N):
    #     res = np.zeros(userCount)
    #     cnt = 0
    #
    #     for k in range(userCount):
    #         sort_predict_item = np.argsort(UI_matrix_predict[k])[::-1]
    #         if np.sum(testratings[:, 0] == k + 1) == 0:
    #             res[k] = 0
    #             cnt += 1
    #             continue
    #         else:
    #             test_item = testratings[testratings[:, 0] == k + 1, 1].flatten()
    #             testlength = len(test_item)
    #
    #         Same_item = len(np.intersect1d(sort_predict_item[:N], test_item))
    #         res[k] = Same_item / testlength
    #
    #     resfinal = np.sum(res) / (userCount - cnt)
    #
    #     return resfinal

    #
if __name__ == '__main__':
    rating_triple = io.loadmat(r'rating.mat')
    # mat文件里可能有多个cell，各对应着一个dataset
    # 可以用keys方法查看cell的名字, 现在要用list(mat.keys()),
    # 另外，读取要用data = mat.get('名字'), 然后可以再用Numpy转为array
    # print(mat.keys())
    # # 可以用values方法查看各个cell的信息
    # print(mat.values())
    rating_triple = rating_triple['ratings']
    # print(rating_triple)
    # print(type(rating_triple))
    rating = np.array(rating_triple)
    # print(type(rating))
    # print(rating)

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

    np.random.seed(0)  # for reproducibility

    feiling_shequnei = io.loadmat(r'E:\code_project\MTLREC_MAIN\community_ciao.mat')
    # mat文件里可能有多个cell，各对应着一个dataset
    # 可以用keys方法查看cell的名字, 现在要用list(mat.keys()),
    # 另外，读取要用data = mat.get('名字'), 然后可以再用Numpy转为array
    # print(mat.keys())
    # # 可以用values方法查看各个cell的信息
    # print(mat.values())
    feiling_shequnei = feiling_shequnei['communityMatrix']
    # print(feiling_shequnei)
    # print(type(feiling_shequnei))
    feiling_shequnei = np.array(feiling_shequnei)
    # print(type(feiling_shequnei))
    precommunityMatrix_juzhen = sparse.csr_matrix(feiling_shequnei)
    communityMatrix = precommunityMatrix_juzhen

    user_rating = io.loadmat(r'E:\code_project\MTLREC_MAIN\userrating.mat')
    # mat文件里可能有多个cell，各对应着一个dataset
    # 可以用keys方法查看cell的名字, 现在要用list(mat.keys()),
    # 另外，读取要用data = mat.get('名字'), 然后可以再用Numpy转为array
    # print(mat.keys())
    # # 可以用values方法查看各个cell的信息
    # print(mat.values())
    user_rating = user_rating['userrating']
    # print(feiling_shequnei)
    # print(type(feiling_shequnei))
    userrating_matrix = np.array(user_rating)

    obj = MTLREC(usercount,itemcount)
    obj.train()
