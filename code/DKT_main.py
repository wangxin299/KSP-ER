import torch
# torch.cuda.current_device()
from Dataset import *
from SimpleRNN import *
import os
import sklearn as skl
from sklearn.model_selection import train_test_split
import gc
from torch.utils.data import DataLoader
from utils import *
# import utils
from torch.optim import Adam
import torch.nn as nn
import tqdm
#import torch.nn.functional as F
# from keras.preprocessing import text, sequence
from keras.preprocessing import sequence
import pandas as pd

RANDOMSTATE = 2021
MAXLEN = 8200

if __name__ == '__main__':
    print(torch.cuda.is_available())
    seed_everything(RANDOMSTATE)
    # Assistments_path = '../data/L_DKT.csv'
    Assistments_path='../data/assistment2009/Reformed_skill_builder_data.csv'
    if not os.path.exists(Assistments_path):  #存在就不用进行数据预处理了，Reformed_skill_builder_data.csv是处理后的结果。
        AssistmentsDatasetPreprocessing()
    train = pd.read_csv(Assistments_path,
                        dtype={'user_id': int, 'skill_trace': str, 'correct_trace': str, 'num_of_skill': int})
    print(train.columns)
    print(train['num_of_skill'].describe())
    train = train[train['num_of_skill'] >=1]
    y = train['correct_trace']
    X = train[['skill_trace', 'num_of_skill']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)
    num_of_skill_train = X_train['num_of_skill']
    num_of_skill_test = X_test['num_of_skill']
    X_train = list(X_train.values)
    y_train = list(y_train.values)
    X_test = list(X_test.values)
    y_test = list(y_test.values)

    X_train = list(map(lambda x: eval(x[0]), X_train)) #map(lambda x : x*2, [1, 2, 3, 4]) #Output [2, 4, 6, 8]
    X_test = list(map(lambda x: eval(x[0]), X_test))  #eval() 函数用来执行一个字符串表达式，并返回表达式的值。

    #在python2中map（）函数返回的是一个列表，但是在python3中1他返回的是一个迭代数（iteration）,所以前面加一个list将迭代器转为列表。
    #map() 会根据提供的函数对指定序列做映射。第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。

    t_len = len(y_train)
    for i in range(t_len):
        y_train[i] = eval(y_train[i])
    t_len = len(y_test)
    for i in range(t_len):
        y_test[i] = eval(y_test[i])
    num_of_skills = 124

    for i in range(len(X_train)):
        t_xlist, t_ylist = X_train[i],y_train[i]
        for j in range(len(t_xlist)):
            X_train[i][j] = X_train[i][j] + t_ylist[j]*num_of_skills #这一步公式的意义：  x + y * 29

    for i in range(len(X_test)):
        t_xlist, t_ylist = X_test[i],y_test[i]
        for j in range(len(t_xlist)):
            X_test[i][j] = X_test[i][j] + t_ylist[j]*num_of_skills

    del num_of_skills

    X_train = sequence.pad_sequences(X_train, maxlen=MAXLEN, padding='post')
    y_train = sequence.pad_sequences(y_train, maxlen=MAXLEN, padding='post')
    X_test = sequence.pad_sequences(X_test, maxlen=MAXLEN, padding='post')
    y_test = sequence.pad_sequences(y_test, maxlen=MAXLEN, padding='post')

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    print("训练集的大小",X_train.shape[0])

    del X, y
    gc.collect()  #Python垃圾回收机制:gc模块

    ##########################################################################################
    #
    # train
    #
    ##########################################################################################
    lr = 1e-3
    # EPOCH = 200
    EPOCH = 20
    batch_size = 16
    MASKED = True

    m = SimpleRNN()
    m.cuda()
    #m.cpu()
    opt = Adam(lr=lr, params=m.parameters())
    bce = nn.BCEWithLogitsLoss()#之后可以改动
    # bce = nn.BCELoss()
    m_criterion = Metrics()

    # # 定义优化器
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    # # 指数衰减学习率控制器： 即每个epoch都衰减lr = lr * gamma,即进行指数衰减
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)


    train_dataset = AssistmentsDataset(X_train, y_train, MAXLEN, num_of_skill_train)
    # print("train_dataset[0]：seqx:", train_dataset[0][0].size())
    # print("train_dataset[1]：seqy:", train_dataset[0][1].size())
    # print("train_dataset[2]：seq_y_mask:", train_dataset[0][2].size())
    # print("train_dataset[2]：seq_y_mask:", train_dataset[0][2])    # 该训练集共有3286项学生训练数据
    # print("train_dataset[3]：mask_in_nn:", train_dataset[0][3].size())   # mask_in_nn -> torch.Size([8200, 29])
    # print("train_dataset[3]：mask_in_nn:", train_dataset[0][3])
    # print("train_dataset[3]：mask_in_nn:", train_dataset[0][3])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  #batch_size = 32

    # ###################################################################################
    # # 查看 各个返回值的shape ：
    # seq_x, seq_y, seq_y_mask, mask_in_nn = iter(train_dataloader).next()
    # print("shape:",seq_x.shape, seq_y.shape, seq_y_mask.shape, mask_in_nn.shape)
    # # shape: torch.Size([32, 8200]) torch.Size([32, 8200]) torch.Size([32, 8200]) torch.Size([32, 8200, 29])
    # ###################################################################################

    # # 输出batch数量  103个batch
    # print('len of a train_dataloader batchtensor: ', len(list(iter(train_dataloader))))
    # # 输出一个batch
    # print('one batch train_dataloader: ', iter(train_dataloader).next())


    test_dataset = AssistmentsDataset(X_test, y_test, MAXLEN, num_of_skill_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # # 输出batch数量   26个batch
    # print('len of a test_dataloader batchtensor: ', len(list(iter(test_dataloader))))
    # # 输出一个batch
    # print('one batch test_dataloader: ', iter(test_dataloader).next())

    ################################################################################
    ##############测试
    # cishu = 0
    # for seq_x, seq_y, seq_y_mask, mask_in_nn in tqdm.tqdm(train_dataloader):
    #     cishu  = cishu+1
    #     print("batch的次数：",cishu)                                     #最后一个batch22个不足32
    #     print("第{0}个batch里的数据{1}是 ：".format(cishu,seq_y[:1,:5]))  #总共分了103个batch
    #     print("第{0}个batch里的数据{1}是 ：".format(cishu, seq_y.size())) #可以看到每一个batch大小为32
    ###############################################################################
    # for seq_x, seq_y, seq_y_mask, mask_in_nn in tqdm.tqdm(train_dataloader):  # 共103个batch，每个batch大小为32
    #     print(seq_x.shape, seq_y.shape)                                      #循环103次
    #     print(seq_y_mask.shape, mask_in_nn.shape)
    #     print(type(seq_x),type(seq_y),type(seq_y_mask),type(mask_in_nn))

    # f = open("../result/L_DKT_result.txt", "w")  # 打开文件以便写入
    # f = open("../result/L_DKT_result.txt", "w")  # 打开文件以便写入
    for epoch in range(EPOCH):  #训练200轮
        avg_loss = 0.
        m.train()  # Sets the module in training mode.
        for seq_x, seq_y, seq_y_mask, mask_in_nn in tqdm.tqdm(train_dataloader): #共103个batch，每个batch大小为32
            # print("train_dataloader:",len(train_dataloader))  # train_dataloader: 103
            # print(seq_x.shape, seq_y.shape)                                      #循环103次

            #############################################################################
            # 用CPU进行计算。
            seq_y = seq_y.cuda()
            seq_x = seq_x.cuda()
            seq_y_mask = seq_y_mask.cuda()
            mask_in_nn = mask_in_nn.cuda()
            #############################################################################

            out = m(seq_x)  # 这里的out是logits
            # print("out 1:",out.shape)  # torch.Size([32, 8200, 29])
            # out1 = out
            out = torch.squeeze(out, dim=-1) # [batch,seq_len,numof_skills]  经过该操作out.shape不变
            # print("out 2:", out.shape)   # torch.Size([32, 8200, 29])
            ## print("是否相等:",out1 == out)  # 表面上看一样
            ## print("是否相等:", id(out1) == id(out))  # id 不等
            #解释： b=torch.squeeze(a，N) 就是将a中所有为1的维度删掉，不为1的维度没有影响。

            # print(out.shape)      #torch.Size([32, 8200, 29])
            # print("out:",out)
            # print("1", out[0, 0, :])
            # print("1",out[0,0,:].shape)     #某一时刻的学生知识状态？？？肯定不是  # out[0,0,:].shape 是 1 torch.Size([29])
            # print("2",F.sigmoid(out[0,0,:]))
            # input()
            # print(out[0],seq_x[0],seq_y[0])
            # input()
            # print(mask_in_nn.shape,out.shape,seq_y.shape)  #torch.Size([32, 8200, 29]) torch.Size([32, 8200, 29]) torch.Size([32, 8200])
            if MASKED:
                # print(out.shape,seq_y.shape)
                # print(mask[0])
                # print(mask.shape)
                # print(out.shape)
                # print("out 1:", out)
                # # out.shape -> torch.Size([32, 8200, 29])
                out = out.masked_select(mask=mask_in_nn)      # 只挑选出True的值进行计算。返回一维tensor
                # print("out 2:", out.shape) ## torch.Size([3083])、torch.Size([3786])
                ## out = F.sigmoid(out)  # 查看概率
                ## print("out sigmoid:", out)
                # print(out.shape,out1.shape)   # torch.Size([3083]) torch.Size([32, 8200, 29])
                # print("seq_y 1:", seq_y.shape) # torch.Size([32, 8200])
                seq_y = seq_y.masked_select(mask=seq_y_mask)  # 只挑选出True的值进行计算。返回一维tensor
                # print("seq_y:",seq_y)
                # print("seq_y:", len(seq_y))
                # print("seq_y:",seq_y.shape)     # torch.Size([3083])、torch.Size([3786])
                # print("seq_y 2:", seq_y)
                # print(seq_y.shape,seq_y1.shape)  #torch.Size([3083]) torch.Size([32, 8200])  # shape  不一样
                # print(out, seq_y)
                loss = bce(out, seq_y)
                #print("loss",loss)      # loss tensor(0.6846, grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
                # out = F.sigmoid(out)
                # out = out.view(-1,1)
                # print(out.shape)
                # loss = maskNLLLoss(out,seq_y,mask)
                # print(out.shape, seq_y.shape)  # 二者shape一样。
            else:
                loss = bce(out, seq_y) #采用BECWithLogitsLoss，相当于把BCELoss和sigmoid融合了，也就是省略了sigmoid这个步骤。
                print(out.shape, seq_y.shape)
            # input()
            # loss = bce(out, seq_y)
            # loss,_ = maskNLLLoss(out, seq_y, mask)
            opt.zero_grad()           ## 梯度清零
            loss.backward()           ## 反向传播求解梯度
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(m.parameters(), 20)
            opt.step()                ## 更新权重参数
            avg_loss += loss.item() / len(train_dataloader)
            #print("avg_loss", avg_loss)  #
        print("avg_loss", avg_loss)  # 一个epoch结束，输出一次avg_loss: 一个epoch中，103个batch训练节后后的平均loss。

        m.eval()   # Sets the module in evaluation mode.
        avg_loss = 0.
        for seq_x, seq_y, seq_y_mask, mask_in_nn in tqdm.tqdm(test_dataloader):

            ########################################################
            # 用CPU计算
            seq_y = seq_y.cuda()
            seq_x = seq_x.cuda()
            seq_y_mask = seq_y_mask.cuda()
            mask_in_nn = mask_in_nn.cuda()
            ########################################################

            out = m(seq_x)

            out = torch.squeeze(out, dim=-1)
            if MASKED:
                out = out.masked_select(mask=mask_in_nn)  # out 是预测下一知识点的答对概率的logits。
                # print("out.shape1:", out.shape)
                # print("out1:", out)
                seq_y = seq_y.masked_select(mask=seq_y_mask)
            loss = bce(out, seq_y)
            # print("loss:",loss) # loss: tensor(0.6929, grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
            # print("loss.item():",loss.item())    # 从tensor转化为数字
            # out = F.sigmoid(out)   # 然后经过F.sigmoid() 这个激活函数 算是得到下一知识点的答对概率。
            out = torch.sigmoid(out)   # 此处的out是经过sigmoid（）激活函数 处理后 的预测的概率（0到1之间）。
            # 输出知识掌握程度

            # print("out.shape2:",out.shape)
            # print("out2:", out)
            # out = out.cpu().tolist()    # 转为list格式。
            #
            # # print("out3:", out)
            # seq_y = seq_y.cpu().tolist()  # 转为list格式。
            # 更换GPU
            out = out.cuda().tolist()  # 转为list格式。

            # print("out3:", out)
            seq_y = seq_y.cuda().tolist()  # 转为list格式。

            avg_loss += loss.item() / len(test_dataloader)   # len(test_dataloader)为26，即26个batch，每个batch大小为32.
            # average loss，指这一轮训练的batch中loss总和除以batch的次数。
            # 简单理解:for-loop结束后，一共训练了26轮，然后avg_loss是26轮batch训练后的平均loss值。
            # print("avg_loss:", avg_loss)  # avg_loss: 0.026648601660361655

            m_criterion.step(y_proba=out, y_true=seq_y)   # step():其实就是 传参
        m_criterion.compute()
        # f = open("../result/L_DKT_result.txt", mode='a')  # 打开文件以便写入
        f = open("../result/assistment2009/assistment_DKT_result.txt", mode='a')  # 打开文件以便写入
        print('epoch:', epoch+1, 'auc_score:', m_criterion.avg_auc, 'acc:', m_criterion.avg_acc, 'f1:',
              m_criterion.avg_F1, 'recall', m_criterion.avg_recall,file=f)
        m_criterion.next_epoch() # 对历史评价标准记录进行存储。
        print('Epoch {}/{} \t loss={:.4f} \t'.format(epoch + 1, EPOCH, avg_loss))
        print('\n')
        # print中含有空格 在真正输出的时候也是严格按照空格输出 \t 是一个缩进。
        if epoch % 20 == 0:
            # torch.save(m.state_dict(), '../pkl/L_DKT_sRNN_20.pkl')
            torch.save(m.state_dict(), '../pkl/assistment2009_DKT_sRNN_100.pkl')
    f.close  # 关闭文件