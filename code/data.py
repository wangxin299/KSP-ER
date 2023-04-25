'''
主要是加载涉及到的文件操作
'''
import numpy as np
import pandas as pd
import csv
def get_df_translation():
    df = pd.read_csv('C:/Users/wangxin/Desktop/majian/OLI_data/OLI_translation_noNA.csv',encoding='utf-8')
    return df
def get_df_problem():
    df=pd.read_csv('C:/Users/wangxin/Desktop/majian/OLI_data/AllData_problem_2011F.csv',encoding='utf-8')
    return df
def get_df_skill():
    df=pd.read_csv('C:/Users/wangxin/Desktop/majian/OLI_data/skills.csv',encoding='utf-8')
    return df
#得到所有学生的列表(333)
def get_students():
    df=pd.read_csv('C:/Users/wangxin/Desktop/majian/OLI_data/students.csv',encoding='utf-8')
    students=df['student']
    return students
#包含学生每道题得分的矩阵
def get_df_avg():
    df = pd.read_csv('../data/L_DMF_input.csv', encoding='utf-8')
    return df
#获取知识点掌握情况
def get_kcLevel():
    df=pd.read_csv('../data/assistment2009/assist2009_knowledge_level_matrix.csv',index_col=0)
    return df

#获取某个学生的知识点掌握情况
def getStudentKclever(student_id):
    matric = get_kcLevel().values
    stu_lever = np.array(matric[student_id])
    return stu_lever
#获取关联规则

def get_rule():
    df=pd.read_csv('../data/assistment2009/Apriori_rules.csv')
    return df
#得到矩阵分解后的P矩阵
def get_P():
    df = pd.read_csv('C:/Users/wangxin/Desktop/majian/OLI_data/P.csv')
    P = df.values
    return P
#得到矩阵分解后的Q矩阵
def get_Q():
    df = pd.read_csv('C:/Users/wangxin/Desktop/majian/OLI_data/Q.csv')
    Q = df.values
    return Q
#返回学生列表便于索引
def read_students():
    df=pd.read_csv('C:/Users/wangxin/Desktop/majian/OLI_data/data_plus1.csv')
    students=df['user']
    students=list(set(students))
    return students
#将答题列表转成R矩阵 list->mat
def sequence2mat(sequence, N, M):
    # input:
    # sequence: the list of rating information
    # N: row number, i.e. the number of users
    # M: column number, i.e. the number of items
    # output:
    # mat: user-item rating matrix
    records_array = np.array(sequence)
    mat = np.zeros([N, M])
    row = records_array[:, 0].astype(int)
    col = records_array[:, 1].astype(int)
    values = records_array[:, 2].astype(np.float32)
    mat[row, col] = values

    return mat
#读取答题记录并将其转为字典
# def get_recordToDirt(problem_df):
#     # 以每个学生为单位获取答题情况
#     record = {}
#     for i in range(len(problem_df)):
#         user = problem_df.iloc[i]['Anon Student Id']
#         exercise = problem_df.iloc[i]['Problem Name']
#         result = problem_df.iloc[i]['Avg']
#         if (user in record.keys()):
#             record[user][exercise] = result
#         else:
#             record[user] = {}
#             record[user][exercise] = result
#     return  record
# #得到排序后的习题序列，将其作为索引
# def get_exercise(problem_df):
#     pset = set(problem_df[2])
#     li = list(pset)
#     li.sort()
#     return li
# #得到习题在按字符串顺序排序后的在习题列表中的索引
# def get_index(li,x):
#     index=li.index(x)
#     return index
#将数据处理成学生答题信息矩阵,写入文件
# def write_mat():

#
#     # 改动 添加.values
#     problem_df=get_df_avg()
#         # .values
#     #
#     li=get_exercise(problem_df)
#     mat = np.zeros((6866, 90))
#     record=get_recordToDirt(problem_df)
#     students = list(record.keys())
#     studentsli = read_students()
#     for student in students:
#         if (student in studentsli):
#             exercises = list(record[student].keys())
#             for exercise in exercises:
#                 result = record[student][exercise]
#                 colindex = get_index(li, exercise)
#                 rowindex = studentsli.index(student)
#                 mat[rowindex][colindex] = result
#     mat=pd.DataFrame(mat)
#     mat.to_csv('C:/Users/wangxin/Desktop/majian/OLI_data/Rmat.csv')
def read_mat():
    df=pd.read_csv('C:/Users/wangxin/Desktop/majian/OLI_data/Rmat.csv',index_col=0)
    mat=df.values
    return mat
#将有数据的信息提取出来转换成列表格式[student,exercise,result]，便于矩阵分解
# def get_mfData():
#     #problem_df = get_df_problem()
#
#     # #####改动 添加.values
#     problem_df=get_df_avg()
#         # .values
#
#     li = get_exercise(problem_df)
#     record = get_recordToDirt(problem_df)
#     students = list(record.keys())
#     data = []
#     #需要与知识追踪用到的学生对应起来
#     studentsli=read_students()
#     for student in students:
#         if(student  in studentsli):
#             exercises = list(record[student].keys())
#             for exercise in exercises:
#                 x = []
#                 result = record[student][exercise]
#                 colindex = get_index(li, exercise)
#                 rowindex = get_index(studentsli,student)
#                 if(rowindex==331):
#                     print(student)
#                 # mat[rowindex][colindex]=result
#                 x.append(rowindex)
#                 x.append(colindex)
#                 x.append(result)
#                 data.append(x)
#         else:
#             continue
#     return data
def getSkillIndex(skillname):
    skill_df = get_df_skill()
    skill = {}
    for i in range(len(skill_df)):
        name = skill_df.iloc[i]['skill']
        index = skill_df.iloc[i]['id']
        skill[name] = index
    if skillname in skill.keys():
         index =skill[skillname]
    else:
        index=-1

        #print(skillname)
    return  index
def getSkills():
    skill_df = get_df_skill()
    skill = []
    for i in range(len(skill_df)):
        skill.append(skill_df.iloc[i]['skill'])
    return skill
#得到知识点习题关系矩阵
def get_QC():
    problem_df = get_df_problem()
    f = open('C:/Users/wangxin/Desktop/majian/OLI_data/qc.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    skills = getSkills()
    head = ['exercise']
    exercises = set(problem_df['Problem Name'])
    exercises = list(exercises)
    #对习题按知识点大小排序
    exercises.sort()
    head = head + skills
    csv_writer.writerow(head)
    for exercise in exercises:
        kcs = problem_df[problem_df['Problem Name'] == exercise].iloc[0]['KC List (F2011)']
        a = kcs.split(', ')
        arr = np.zeros(80)
        for kc in a:
            index = getSkillIndex(kc)
            if (index > -1):
                arr[index] = 1
        write = []
        write.append(exercise)
        num = list(arr)
        write = write + num
        csv_writer.writerow(write)
def get_wrong_list():
    df = pd.read_csv("../data/L_DKT.csv")
    df = df.loc[0, "skill_trace"]
    # df[0]
    import ast

    # l = "['1', '2', 'hello', 'str', 1.2]"
    df = ast.literal_eval(df)
    # print(type(my_list))
    # print(my_list)
    df[89]
    sq = pd.read_csv(r"C:\Users\wangxin\Desktop\研一\代码\StudyRecommend\files\student_answer_records.csv",
                     header=None).to_numpy()
    dirt = {}
    for i in range(sq.shape[1]):
        user = i + 1
        for j in range(sq.shape[0]):
            # print(j)
            kc = df[j]
            result = sq[j][i]

            # print(kc)
            if (result == 0):
                dirt.setdefault(user, []).append(kc)
    wro_list = []
    for key in dirt.keys():
        li = set(dirt[key])
        li = list(li)
        wro_list.append(li)
    # print(wro_list)
    return wro_list
#得到习题难度,所有答题记录的统计学生难度
def get_exercise_hardness():
    df = get_df_avg()
    dirtvalue = {}
    dirtcnt = {}
    for i in range(len(df)):
        problem = df.iloc[i]['Problem Name']
        result = df.iloc[i]['Avg']
        if (problem in dirtvalue.keys()):
            dirtvalue[problem] = dirtvalue[problem] + result
            dirtcnt[problem] += 1
        else:
            dirtvalue[problem] = result
            dirtcnt[problem] = 1
    dirt={}
    for key in dirtvalue.keys():
        value = dirtvalue[key]
        cnt = dirtcnt[key]
        dirt[key]=value/cnt
    return dirt
    #写入csv
    # f = open('C:/Users/wangxin/Desktop/majian/OLI_data/hardness.csv', 'w', encoding='utf-8', newline='')
    # csv_writer = csv.writer(f)
    # head = ['exercise', 'hardness', 'type']
    # csv_writer.writerow(head)
    # for key in dirt.keys():
    #     hard = dirt[key]
    #     if (hard < 0.5):
    #         type = 'diff'
    #     elif (hard > 0.75):
    #         type = 'easy'
    #     else:
    #         type = 'mid'
    #     csvrow = [key, hard, type]
    #     csv_writer.writerow(csvrow)
    # f.close()
#得到每个习题的难度
def read_hardness():
    df=pd.read_csv('C:/Users/wangxin/Desktop/majian/OLI_data/hardness.csv')
    return df
#得到学生答题列表，每个学生作答过的所有题目，包括学生做过的相同数据集
def get_studentExercises():
    student = list(get_students())
    df = get_df_problem()
    li = []
    for i in range(len(student)):
        li.append([])
    for i in range(len(df)):
        user = df.iloc[i]['Anon Student Id']
        exercise = df.iloc[i]['Problem Name']
        id = student.index(user)
        li[id].append(exercise)
    return li
def percent(fenzi,fenmu):
    if fenmu==0:
        return 0
    else:
        return fenzi/fenmu
#计算学生答题偏好，将结果写入文档
# def get_studentPerference():
#     record=get_recordToDirt(get_df_avg())
#     hard=get_exercise_hardness()
#     f=open('C:/Users/wangxin/Desktop/majian/OLI_data/perference.csv','w',encoding='utf-8',newline='')
#     header=['student','nd','nm','ne','nd_','nm_','ne_','rad','ram','rae','rid','rim','rie']
#     writer=csv.writer(f)
#     writer.writerow(header)
#     students = get_students()
#     for key in students:
#         nd = 0
#         nm = 0
#         ne = 0
#         nd_ = 0
#         nm_ = 0
#         ne_ = 0
#         dirt = record[key]
#         for exercise in dirt.keys():
#             hardness = hard[exercise]
#             result = dirt[exercise]
#             if (hardness < 0.5):
#                 nd += 1
#                 if (result > 0.75):
#                     nd_ += 1
#             elif (hardness == 0.75 or hardness > 0.75):
#                 ne += 1
#                 if (result > 0.75):
#                     ne_ += 1
#             else:
#                 nm += 1
#                 if (result > 0.75):
#                     nm_ += 1
#         a=nd+nm+ne
#         rad=percent(nd,a)
#         ram=percent(nm,a)
#         rae=percent(ne,a)
#         rid=percent(nd_,nd)
#         rim=percent(nm_,nm)
#         rie=percent(ne_,ne)
#         li=[key, nd, nm, ne, nd_, nm_, ne_,rad,ram,rae,rid,rim,rie]
#         writer.writerow(li)
# if __name__ == '__main__':
#    get_studentPerference()
