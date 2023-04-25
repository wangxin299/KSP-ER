'''
挖掘薄弱知识点
'''
import pandas as pd
import numpy as np
# from data import get_kcLevel,getSkills,get_rule
from data import get_kcLevel,get_rule
#得到薄弱知识点
def get_weak_kc(id):
    li=[]
    df=get_kcLevel()
    value=df.values
    #print(row,value)
    #输出知识点掌握情况
    #print(value[id])
    for i in range(0,124):
        if(value[id][i]<0.4):
            li.append(i)

    return li
#按照关联规则薄弱知识点进行扩充
def add_kc(li):
    df=get_rule()
    for concept in li:

        for i in range(len(df)):
            if(df.iloc[i]['concept1']==concept):
                concept2=df.iloc[i]['concept2']
                if (concept2 not in li):
                    li.append(int(concept2))
    return li
#对知识点序列进行排序-得到学习路径
def kc_sort(kc,id):
    #定义前提知识点，以出现的频率排序

    # basic=[8,12,27,28,38,41,29,34]
    # matric = get_kcLevel().values
   
    # print(weak)
    # 对知识点掌握程度进行降序排序

    sorted_index = sorted(weak,key=(lambda x: x[1]),reverse=True)


    # print("sorted_index:",sorted_index)
    sorted_kc=[]#排序后的知识点
    '''
    考虑最后的知识
    '''
    #首先添加前提知识点
    # for i in basic:
    #     if(i in kc):
    #         sorted_kc.append(i)
    for i in sorted_index:
        # if(kc[i] in basic):
        #     continue
        # print("i:",i)
        sorted_kc.append(int(i[0]))
    return sorted_kc
import tqdm
if __name__ == '__main__':
    skill_all = []
    for i in tqdm.tqdm(range(0,4217)):
        skill_id=[]
        # with open('../result/assistment2009/weak_concept_sort.txt',mode='a',encoding='utf-8') as f:
        # # li=get_weak_kc(2)
        # print("学生id：",i)
        # print('薄弱知识点集合:',get_weak_kc(i))
        # # # li=add_kc(li)
        # print('关联后的知识点集合:',add_kc(get_weak_kc(i)))
        # # # li=kc_sort(li,2)
        # print("根据知识点之间的关系进行排序之后的知识点序列",kc_sort(add_kc(get_weak_kc(i)),i))
        skill_id.append(kc_sort(add_kc(get_weak_kc(i)),i))
        skill=[]
        print(skill_id)
        for i in skill_id[0]:
            skill.append(i)
        skill_all.append(skill)



    # skill_all=pd.DataFrame(skill_all)
    # skill_all.to_csv(r'C:\Users\wangxin\Desktop\remmend_DKT_Apriori_DMF\result\assistment2009\assist2009_recommend_skill.csv',index=False)
    #     # print(li)
