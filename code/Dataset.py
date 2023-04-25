import pandas as pd
import numpy as np
import os
import sklearn as skl
from sklearn.preprocessing import LabelEncoder
import tqdm
import torch
from torch.utils.data import Dataset


def AssistmentsDatasetPreprocessing():
    """
    must run this function before Using Assistments Dataset
    :return: Create  a  .csv file in disk
    """
    # train = pd.read_csv('../data/L_DKT.csv')
    train = pd.read_csv('../data/assistment2009/Reformed_skill_builder_data.csv')
    # train=train.copy()
    useful_features = ['user_id', 'skill_id', 'correct_id']
    train = train[useful_features]
    lbcdr = LabelEncoder()
    lbcdr.fit(train['user_id'])
    train['user_id'] = lbcdr.transform(train['user_id'])
    train['skill_id'] = train['skill_id'].fillna(-1)
    lbcdr.fit(train['skill_id'])


    train['skill_id'] = lbcdr.transform(train['skill_id'])
    train['skill_id'] = train['skill_id'].apply(lambda x: x + 1)
    tmp_list = train['user_id'].unique()
    user_trace_dict = dict()
    for i in tmp_list:
        user_trace_dict[i] = [list(), list(), 0]
    nrow = train.shape[0]
    tmp_user_col = train['user_id']
    tmp_skill_col = train['skill_id']
    tmp_correct_col = train['correct']
    for i in range(nrow):
        tmp_user_id = tmp_user_col.iloc[i]
        tmp_skill_id = tmp_skill_col.iloc[i]
        tmp_correct = tmp_correct_col.iloc[i]
        user_trace_dict[tmp_user_id][0].append(tmp_skill_id)
        user_trace_dict[tmp_user_id][1].append(tmp_correct)
    keys = list(user_trace_dict.keys())
    Reformed_DataFrame = pd.DataFrame(columns=['user_id', 'skill_trace', 'correct_trace', 'num_of_skill'])
    for uid in tqdm.tqdm(keys):
        row = user_trace_dict[uid]
        # dic = {'user_id':uid,'skill_trace':str(row[0]),'correct_trace':str(row(1))}
        Reformed_DataFrame.loc[Reformed_DataFrame.shape[0] + 1] = {'user_id': uid, 'skill_trace': str(row[0]),
                                                                   'correct_trace': str(row[1]),
                                                                   'num_of_skill': len(row[0])}

        # new = pd.DataFrame([str(uid),str(row[0]),str(row[1])],columns=['user_id','skill_trace','correct_trace'])
        # Reformed_DataFrame = Reformed_DataFrame.append(new)
    Reformed_DataFrame.to_csv('../result/assistment2009/assistment2009_result.csv',
                              index=False)
    print("save file")
    return


class AssistmentsDataset(Dataset):
    def __init__(self, X, y, max_len, num_of_skill, max_skill=124):
        self.X = X
        self.y = y
        self.num_of_skill = num_of_skill   #
        self.max_len = max_len    #8200
        self.max_skill = max_skill   #29
        # self.max_len = max_len
        # train_x = pad_sequences(train_x, maxlen=self.maxlen)
        # pd.read_csv(data_path, dtype={'user_id': np.int8, 'skill_trace': str, 'correct_trace': str})

    def __getitem__(self, item):   #根据索引进行dataframe里某一行数据的操作。
        seqx = self.X.iloc[item]  #iloc指定DataFrame的第item-1列  也就是某个人的答题序列
        seqy = self.y.iloc[item]  #这个是某个人的答题结果序列
        numofskill = self.num_of_skill.iloc[item]  #某个人答了多少个题。
        numofskill = min(self.max_len, numofskill)  #这一步设定最大长度是 max_len。 8200
        # seq_x 中 0 代表padding 其他：代表习题类型1-29为作错，125-248为正确
        # seq_x 保留为[:seq_x-1]
        # seq_y 保留为[1:seq_y]
        # 以上说明的都是有意义的

        tmp_x = list(seqx)
        next_x = tmp_x.copy() # 'list' object has no attribute 'deepcopy' 这里虽然是copy但是是 深复制 。
        next_x.pop(0)   # POP出第一个元素   这里next_x用于mask。
        next_x.append(0)  # 在最后添加一个padding 0。
        tmp_x[numofskill-1] = 0  # 这一步的操作是：seq_x 保留为[:seq_x-1] 把最后一个skill置零
        tmp_y = list(seqy)
        tmp_y.pop(0)
        tmp_y.append(0)

        # 两个有意义长度都为 numofskill-1 实际长度为max_len
        # for skill_id in seqx
        # print(type(mask_in_nn), mask_in_nn.shape)

        seqx = torch.tensor(tmp_x, dtype=torch.long).contiguous()
        seqy = torch.tensor(tmp_y, dtype=torch.float).contiguous()
        # torch.BoolTensor( )
        seq_y_mask = torch.zeros(self.max_len, dtype=torch.bool)  #创建bool（False）类型的tensor，大小为8200。
        # mask = torch.BoolTensor(self.max_len)

        seq_y_mask[:numofskill-1] = True #将前 numofskill-1 个skill置为True。 前闭后开

        # 有意义的seq长度为seqx-1 （0，numofskill-1）前闭后开。
        mask_in_nn = np.zeros(shape=(self.max_len, self.max_skill))
        for i in range(numofskill-1):  # 只取前numofskill-1个答题情况
            mask_in_nn[i, (next_x[i]) % self.max_skill] = 1  # 不进行 -1 操作，实验目的：验证知识为了找出和标签同样个数的位置进行loss计算，至于什么位置没有关系。
            # mask_in_nn[i, (next_x[i] - 1) % self.max_skill] = 1   # 原本是有-1操作的。

        mask_in_nn = torch.tensor(mask_in_nn, dtype=torch.bool)
        #print(seqx.shape, seqy.shape, seq_y_mask.shape, mask_in_nn.shape)
        return seqx, seqy, seq_y_mask, mask_in_nn

    def __len__(self):
        return self.X.shape[0]
