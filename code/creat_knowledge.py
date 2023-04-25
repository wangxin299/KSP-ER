import torch
import torch.nn as nn
from utils import *
from SimpleRNN import *
from Dataset import *
from DKT_main import RANDOMSTATE,MAXLEN
from keras.preprocessing import sequence
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
class T_DataSet(Dataset):
    def __init__(self, X, y, max_len, num_of_skill, max_skill=81):
        super(T_DataSet, self).__init__()
        self.X = X
        self.y = y
        self.num_of_skill = num_of_skill
        self.max_len = max_len
        self.max_skill = max_skill
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        seqx = self.X.iloc[item]
        seqy = self.y.iloc[item]
        numofskill = self.num_of_skill.iloc[item]
        numofskill = min(self.max_len, numofskill)
        tmp_x = list(seqx)
        tmp_y = list(seqy)
        for j in range(len(tmp_x)):
            tmp_x[j] = tmp_x[j] + tmp_y[j] * self.max_skill
        seqx = torch.tensor(tmp_x, dtype=torch.long)
        t_mask = torch.zeros(self.max_len, dtype=torch.bool)
        t_mask[:numofskill] = True
        seqx = seqx.masked_select(t_mask)
        return seqx
NUMOFSKILLS = 124
if __name__ == '__main__':
    print(torch.cuda.is_available())
    seed_everything(RANDOMSTATE)
    Assistments_path = '../data/assistment2009/Reformed_skill_builder_data.csv'
    if not os.path.exists(Assistments_path):
        AssistmentsDatasetPreprocessing()
    train = pd.read_csv(Assistments_path,
     dtype={'user_id': int, 'skill_trace': str, 'correct_trace': str, 'num_of_skill': int})
    # print(train.columns)
    # print(train['num_of_skill'].describe())
    train = train[train['num_of_skill'] >= 1]
    y = train["correct_trace"]
    # print("y:",y.shape)
    # print(y)
    X = train[['skill_trace', 'num_of_skill']]
    # print("X:", X.shape)
    # print(X)
    num_of_skill_train = X['num_of_skill']
    X = list(X.values)
    #提取学生的答题序列
    X = list(map(lambda x: eval(x[0]), X))

    t_len = len(y)
    # print("len(y):",len(y))     # 6866
    for i in range(t_len):
        y[i] = eval(y[i])
    X = sequence.pad_sequences(X, maxlen=MAXLEN, padding='post')
    y = sequence.pad_sequences(y,maxlen=MAXLEN,padding='post')
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    # print("y:",y.shape)
    # print(y)
    # print("X:", X.shape)
    # print(X)
    t_data = T_DataSet(X, y,MAXLEN, num_of_skill_train)
    # print("t_data:",t_data)
    t_dataloader = DataLoader(t_data, batch_size=1, shuffle=False)
    # print("t_dataloader:", t_dataloader)
    ##############################
    model = SimpleRNN()
    model.load_state_dict(torch.load('../pkl/assistment2009_DKT_sRNN_100.pkl'))
    knowledge_level_matrix = np.zeros(shape=(len(X), NUMOFSKILLS))
    # print("knowledge_level_matrix:",knowledge_level_matrix.shape)
    ##############################
    model.eval()
    #同时获取索引值和元素内容
    for idx, seq_x in enumerate(t_dataloader):
        if(idx<5):

            print(seq_x)
        out = model(seq_x)
        # print(out.shape)
        # 对数据维度进行解压
        out = torch.squeeze(out, dim=0)
        #使用sigmoid函数将其映射到0-1区间
        last_level = F.sigmoid(out[-1, :])
        #去掉梯度,但数据内容还是不变得
        knowledge_level_matrix[idx,:] = last_level.detach().numpy()

#np.save("knowledge_level_matrix.npy", knowledge_level_matrix)
print(knowledge_level_matrix.shape)
pd_data = pd.DataFrame(knowledge_level_matrix)
print(pd_data)
pd_data.to_csv('../data/assistment2009/assist2009_knowledge_level_matrix.csv')
