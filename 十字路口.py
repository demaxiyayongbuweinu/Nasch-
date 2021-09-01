# encoding=utf-8
import matplotlib.pyplot as plt
import math
import random
from scipy import stats
import numpy as np
from pandas.core.frame import DataFrame
# from pylab import *                                 #支持中文
from pylab import mpl
import pandas as pd
from scipy import interpolate
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


class car():
    def __init__(self,Location,speed,lane):
        self.Location = Location  ##车辆位置
        self.speed = speed  ##车辆速度
        self.lane=lane##车辆车道


'''生成最初的车辆'''
def Generate_car(road_length,rho,v_max,space1, space2):
    car1 = {} #存储车道1车辆信息的地方
    car2 = {}  # 存储车道1车辆信息的地方
    #生成车辆信息
    allcarnum=int(road_length*rho)
    for i in range(allcarnum):
        if i < allcarnum/2:#分配到第一车道
            lane=1
            car1,space1 = add_car(i, car1, v_max, lane, road_length, space1)
        else:#分配到第二车道
            lane = 2
            car2,space2 = add_car(i, car2, v_max, lane, road_length, space2)
    return car1,car2,space1, space2

def add_car(key,cardict,v_max,lane,road_length,space):
    speed=random.randrange(0,v_max,1)
    location = random.randrange(0, road_length, 1)
    while space[location]==1 or location==len(space)//2:
        location=random.randrange(0,road_length,1)

    cardict[key]=car(location,speed,lane)
    space[location]=1
    return cardict,space


'''生成最初的道路'''
def Generate_road(road_length):
    space1 = [0 for x in range(road_length)] #车道1空间
    space2 = [0 for x in range(road_length)] #车道2空间
    return space1,space2


'''Nasch车辆跟随'''
def Nasch_Follow(car1, car2,space1,space2,v_max, p_slowdown,flownum1,flownum2):
    #变化
    car1, space1=follow(car1, v_max, p_slowdown, space1)
    car2, space2 = follow(car2, v_max, p_slowdown, space2)
    ##位置更新
    car1, space1,flownum1=location_updata(car1, space1,flownum1)
    car2, space2,flownum2 = location_updata(car2, space2,flownum2)

    return car1, car2,space1,space2,flownum1,flownum2

def follow(car,v_max,p_slowdown,space):

    for key,values in car.items():
        # 加速
        values.speed=min(values.speed+1,v_max)
        # 减速
        values.speed=min(values.speed,get_empty_front(values.Location, space))
        # 随机慢化
        if random.random() <= p_slowdown:
            values.speed = max(values.speed - 1, 0)


    return car,space

# 函数：获取和前车的距离
def get_empty_front(location, space):
    # link2 = link * 2   # 周期性边界
    num = 0; i = 1
    temp=location + i
    if int(temp) <= len(space) - 1:
        temp = temp
    else:
        temp = temp - len(space)
    while (space[temp]) == 0:
        # print(1, len(space), location, temp, i)
        num += 1
        i += 1
        temp = location + i
        # print(temp,num,int(temp)<=len(space)-1,temp-len(space))
        if int(temp)<=len(space)-1:
            temp=temp
        else:
            temp -= len(space)
        # print(2,temp,len(space),location)
        if num>=len(space)-1:
            break


    # if location + i<=len(space)-1:
    #
    # else:
    #     while (space[location + i-len(space)]) == 0:
    #         print(2,len(space), location, location + i, i)
    #         num += 1; i += 1
    return num


def location_updata(car,space,flownum):
    for key, values in car.items():

        linshi=values.Location+values.speed
        # if linshi>=len(space):
        #     linshi=linshi-len(space)
        # if space[linshi]==1 and linshi!=values.Location:
        #     print('撞车啦',linshi)

        if values.Location<=10 and linshi>10:
            flownum+=1

        space[values.Location]=0##原来的道路取消
        values.Location=values.Location+values.speed##位置更新
        if values.Location>=len(space):
            values.Location=values.Location-len(space)
        space[values.Location]=1##新的位置道路加载


    return car,space,flownum

'''判断红绿灯'''

def Traffic_light_status(space1,space2,model,time,light_time,model_traffic_light1,model_traffic_light2,car1, car2):
    if model==0: #智能红绿灯开启
        # print('智能红绿灯关闭')
        if light_time==0:
            # print('无红绿灯')
            traffic_light=[1,1]##红绿灯状态  0 红灯 1 绿灯
        else:
            # print('有红绿灯')
            if (time//(light_time/2))%2==0:
                traffic_light=[1,0]##红绿灯状态  0 红灯 1 绿灯
            else:
                traffic_light = [0,1]##红绿灯状态  0 红灯 1 绿灯

    elif model==1: #智能红绿灯开启
        print('智能红绿灯开启')
        ##判断车流密度
        density1=sum(space1[:len(space1)//2])//(len(space1)//2)
        density2 = sum(space2[len(space1) // 2+1:]) // (len(space1) // 2)
        if density1>=density2:
            if sum(model_traffic_light1[-3:])==3:
                traffic_light = [0, 1]  ##红绿灯状态  0 红灯 1 绿灯
            else:
                traffic_light = [1, 0]  ##红绿灯状态  0 红灯 1 绿灯
        else:
            if sum(model_traffic_light2[-3:])==3:
                traffic_light = [1, 0]  ##红绿灯状态  0 红灯 1 绿灯
            else:
                traffic_light = [0,1]  ##红绿灯状态  0 红灯 1 绿灯

    ##设置道路红绿灯状态
    if traffic_light==[1,0]:#主干道绿灯
        space1[len(space1)//2]=0#
        space2[len(space2)//2] = 1#
        model_traffic_light1+=[1]##红绿灯状态记录  0 红灯 1 绿灯
        model_traffic_light2 += [0]
    elif traffic_light==[0,1]:#次干道绿灯
        space1[len(space1) // 2] = 1  #
        space2[len(space2) // 2] = 0  #
        model_traffic_light1 += [0]
        model_traffic_light2 += [1]

    elif traffic_light == [1, 1]:#无绿灯
        traffic_light=Passing_traffic_lights(car1,car2,space1,space2)##判断无红绿灯状态下路口的通行情况
        if traffic_light == [1, 0]:  # 主干道绿灯
            space1[len(space1) // 2] = 0  #
            space2[len(space2) // 2] = 1  #
            model_traffic_light1 += [1]
            model_traffic_light2 += [0]
        elif traffic_light == [0, 1]:  # 次干道绿灯
            space1[len(space1) // 2] = 1  #
            space2[len(space2) // 2] = 0  #
            model_traffic_light1 += [0]
            model_traffic_light2 += [1]

    return space1,space2,model_traffic_light1,model_traffic_light2

def Passing_traffic_lights(car1,car2,space1,space2):
    judge=[]
    road1car2intersection = [] ##道路1下一时刻到达路口的车的信息  位置  速度
    road2car2intersection = [] ##道路1下一时刻到达路口的车的信息  位置  速度
    for key,values in car1.item():
        if values.Location+values.speed>=len(space1)//2 and  values.Location<len(space1)//2:
            road1car2intersection+=[values.Location,values.speed]
    for key,values in car2.item():
        if values.Location+values.speed>=len(space2)//2 and  values.Location<len(space2)//2:
            road2car2intersection+=[values.Location,values.speed]


    if road1car2intersection[0] > road2car2intersection[1]:##主车道的车位置比较靠近路口
        judge=[1,0]##红绿灯状态  0 红灯 1 绿灯
    elif road1car2intersection[0] < road2car2intersection[1]:##次车道的车位置比较靠近路口
        judge = [0, 1]  ##红绿灯状态  0 红灯 1 绿灯
    elif road1car2intersection==[] and  road2car2intersection!=[]:##次车道的车下一时刻到达路口
        judge = [0, 1]  ##红绿灯状态  0 红灯 1 绿灯
    elif road2car2intersection==[] and  road1car2intersection!=[]:##主车道的车下一时刻到达路口
        judge = [1, 0]  ##红绿灯状态  0 红灯 1 绿灯
    else:
        if random.random()>0.5:
            judge = [1, 0]  ##红绿灯状态  0 红灯 1 绿灯
        else:
            judge = [0, 1]  ##红绿灯状态  0 红灯 1 绿灯
    return judge

def Real_time_drawing(model_traffic_light1,model_traffic_light2,space1,space2,road_length,time):
    guding1 = [];
    guding3 = []
    guding2 = [];
    guding4 = []

    for i, m in enumerate(space1):
        if m == 1:
            if i<len(space1)//2:

                guding1 += [road_length // 2]
                guding3 += [i]
            else:
                guding3 += [road_length // 2]
                guding1 += [i]
    for i, m in enumerate(space2):
        if m == 1:
            if i<len(space1)//2:
                guding2 += [road_length // 2]
                guding4 += [i]
            else:
                guding4 += [road_length // 2]
                guding2 += [i]

    plt.plot(guding3, guding1, 'or', markersize=5)
    plt.plot(guding2, guding4, 'sk', markersize=5)

    plt.xlim([0, len(space1) + 1])
    plt.ylim([0, len(space1) + 1])
    tishi = '路口状态为'
    ##红绿灯状态记录  0 红灯 1 绿灯
    if model_traffic_light1[-1] == 0:
        tishi = tishi + '主干道为红灯   '
    else:
        tishi = tishi + '主干道为绿灯   '
    if model_traffic_light2[-1] == 0:
        tishi = tishi + '次干道为红灯'
    else:
        tishi = tishi + '次干道为绿灯'
    plt.xlabel('timestep:' + str(time) + ' \n' + tishi)
    plt.pause(0.1)

    plt.cla()

    return


def statistics(car1, car2, space1, space2):
    memory1 = [0 for i in range(len(space1))]  # 道路1位置存储位置   里面是0  1    0 没有车    1有车
    memory2 = [0 for i in range(len(space1))]  # 道路2位置存储位置   里面是0  1    0 没有车    1有车

    memory1_v = ['.' for i in range(len(space1))]  # 道路1位置存储速度  里面是速度值012345  如果没有车为.
    memory2_v = ['.' for i in range(len(space1))]  # 道路2位置存储速度  里面是速度值012345  如果没有车为.
    for key,values in car1.items():
        if values.Location<=len(space1)//2:
            memory1[values.Location]=1
            memory1_v[values.Location]=values.speed
        else:
            memory2[values.Location] = 1
            memory2_v[values.Location] = values.speed
    for key,values in car2.items():
        if values.Location<=len(space1)//2:
            memory2[values.Location]=1
            memory2_v[values.Location]=values.speed
        else:
            memory1[values.Location] = 1
            memory1_v[values.Location] = values.speed

    return memory1, memory2, memory1_v, memory2_v

def statistics_ana(dict1,dict2):

    return

def plot1(memory1,memory2):
    ##道路时空图   横轴为cell  数轴为times
    for i,m in enumerate(memory1):
        # print(m)
        for j,n in enumerate(m):
            # print(m,n)
            if n==1:
                plt.plot(j,len(memory1)-1-i,'black', marker= '.')
                # hold on
    plt.xlabel('车道')
    plt.ylabel('时间步长')
    ykedu=[str(x) for x in range(len(memory1),-1,-20)]
    ykedu = [str(x) for x in range(0, len(memory1)+1, 20)]
    print(ykedu)
    plt.yticks(ykedu)
    plt.title('车道1时空图')
    plt.savefig('车道1时空图.png')
    plt.show()
    for i, m in enumerate(memory2):
        # print(m)
        for j, n in enumerate(m):
            if n == 1:
                plt.plot(j, len(memory2)-1-i, 'black', marker='.')
                # hold on
    plt.xlabel('车道')
    plt.ylabel('时间步长')
    ykedu = [x for x in range(len(memory2), 0, -10)]
    plt.yticks(ykedu)
    plt.title('车道2时空图')
    plt.savefig('车道2时空图.png')
    plt.show()
    return

def plot2(memory1_v,memory2_v):
    ##速度的时空图
    # fig = plt.figure(figsize=(16, 8), facecolor='black')
    fig, ax = plt.subplots()
    style = dict(size=5, color='black')
    for i,m in enumerate(memory1_v):
        # print(m)
        for j,n in enumerate(m):
            ax.text(j,len(memory1_v)-1-i,n,style)
            # print(j,i,n)
    # ax.text(10, 12, '7')
    # print(len(memory1_v)+1)
    plt.xlim([0,len(memory1_v[0])+1])
    plt.ylim([0,len(memory1_v)+1])
    plt.xlabel('车道1位置')
    plt.ylabel('时间')
    plt.title('车道1速度图')
    plt.savefig('车道1速度图.png')
    plt.show()

    fig, ax = plt.subplots()
    style = dict(size=5, color='black')
    for i, m in enumerate(memory2_v):
        # print(m)
        for j, n in enumerate(m):
            ax.text(j, len(memory2_v)-1-i, n, style)
            # print(j,i,n)
    # ax.text(10, 12, '7')
    # print(len(memory1_v)+1)
    plt.xlim([0, len(memory2_v[0]) + 1])
    plt.ylim([0, len(memory2_v) + 1])
    plt.xlabel('车道2位置')
    plt.ylabel('时间')
    plt.title('车道2速度图')
    plt.savefig('车道2速度图.png')
    plt.show()
    return

def plot3():
    # configuration-x图
    x=[]
    pltmodel = 3
    p = [0.2, 0.5, 0.8]  ##一定为偶数
    road1 = []
    road2 = []
    for i in p:
        # density+=[i/100]
        print('随机慢化概率为%d' % (i))
        memory1, memory2,total_times = main(pltmodel, i)
        x,y1,y2=plot3_ana(memory1,memory2,total_times)

        road1 += [y1]
        road2 += [y2]

    # print(len(x),len(y1),y1)
    plt.plot(x, road1[0][0], 'y-')
    plt.plot(x, road1[1][0], 'b--')
    plt.plot(x, road1[2][0], 'r--')
    plt.xlabel('元胞位置')
    plt.ylabel('configuration')
    plt.title('车道1 configuration-x')
    plt.legend(p)
    plt.savefig('车道1 configuration-x.png')
    plt.show()

    plt.plot(x, road2[0][0], 'y-')
    plt.plot(x, road2[1][0], 'b--')
    plt.plot(x, road2[2][0], 'r--')
    plt.xlabel('元胞位置')
    plt.ylabel('configuration')
    plt.title('车道2 configuration-x')
    plt.legend(p)
    plt.savefig('车道2 configuration-x.png')
    plt.show()
    road3=[]

    for i,m in enumerate(road1):
        temp=[]
        for j,b in enumerate(m[0]):
            temp+=[(road1[i][0][j]+road2[i][0][j])/2]
        road3+=[temp]
    plt.plot(x, road3[0], 'y-')
    plt.plot(x, road3[1], 'b--')
    plt.plot(x, road3[2], 'r--')
    plt.xlabel('元胞位置')
    plt.ylabel('configuration')
    plt.title('总车道 configuration-x')
    plt.legend(p)
    plt.savefig('总车道 configuration-x.png')
    plt.show()
    return

def plot3_ana(memory1,memory2,total_times):
    y1=[]
    y2=[]
    x=[]
    mm1=np.mat(memory1)
    mm2 = np.mat(memory2)
    # print(mm1)
    y1=np.sum(mm1, axis=0)/total_times
    y2 = np.sum(mm2, axis=0)/total_times
    # print(y1)
    for i,m in enumerate(memory1):
        if i==0:
            for j,n in enumerate(m):
                x+=[(j+1)/len(m)]
    y1=y1.tolist()
    y2 = y2.tolist()
    return x,y1,y2


def plot4():
    #流量密度图
    pltmodel=4
    density=[]
    flow1=[]
    flow2=[]

    for i in range(1,101):
        density+=[i/100]
        print('车辆密度为百分之%d' %(i))
        if i==100:
            flow1 += [0]
            flow2 += [0]
        else:
            flow_temp1,flow_temp2=main(pltmodel,i/100)
            flow1 += [flow_temp1]
            flow2 += [flow_temp2]
    plt.scatter(density,flow1,c='b',marker='o')
    plt.xlabel('密度')
    plt.ylabel('车道1流量')
    plt.title('车道1 current-density')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('车道1 current-density.png')
    plt.show()

    plt.scatter(density, flow2, c='b', marker='o')
    plt.xlabel('密度')
    plt.ylabel('车道2流量')
    plt.title('车道2 current-density')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('车道2 current-density.png')
    plt.show()

    flow3 = []
    for i, m in enumerate(flow1):

        temp = [(flow1[i] + flow2[i])/2]
        flow3 += [temp]
    plt.scatter(density, flow3, c='b', marker='o')
    plt.xlabel('密度')
    plt.ylabel('总车道流量')
    plt.title('总车道 current-density')
    plt.savefig('总车道 current-density.png')
    plt.show()
    return

def plot5():
    pltmodel=5
    T=[10,20,30,40,50,60]##一定为偶数
    T=[x for x in range(10,60,4)]
    print(T)
    flow1 = []
    flow2 = []
    for i in T:
        # density+=[i/100]
        print('红绿灯时长为%d秒' % (i))
        flow_temp1,flow_temp2=main(pltmodel,i)
        flow1 += [flow_temp1]
        flow2 += [flow_temp2]
    plt.plot(T, flow1)
    plt.xlabel('红绿灯时长')
    plt.ylabel('车道1流量')
    plt.title('车道1红绿灯密度图')
    plt.savefig('车道1红绿灯密度图.png')
    plt.show()

    plt.plot(T, flow2)
    plt.xlabel('红绿灯时长')
    plt.ylabel('车道2流量')
    plt.title('车道2红绿灯密度图')
    plt.savefig('车道2红绿灯密度图.png')
    plt.show()

    flow3 = []
    for i, m in enumerate(flow1):

        temp = [(flow1[i] + flow2[i])/2]
        flow3 += [temp]
    plt.plot(T, flow3)
    plt.xlabel('密度')
    plt.ylabel('总车道流量')
    plt.title('总车道红绿灯密度图')
    plt.savefig('总车道红绿灯密度图.png')
    plt.show()
    return

def plot6():
    #流量密度图
    pltmodel=6
    density=[]
    flow1=[]
    flow2=[]

    for i in range(1,101):
        density+=[i/100]
        print('车辆密度为百分之%d' %(i))
        if i==100:
            flow1 += [0]
            flow2 += [0]
        else:
            flow_temp1,flow_temp2=main(pltmodel,i/100)
            flow1 += [flow_temp1]
            flow2 += [flow_temp2]
    print()
    plt.scatter(density,flow1,c='b',marker='o')
    plt.xlabel('密度')
    plt.ylabel('车道1流量')
    plt.title('车道1流量密度图')
    plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.savefig('车道1流量密度图.png')
    plt.show()

    plt.scatter(density, flow2, c='b', marker='o')
    plt.xlabel('密度')
    plt.ylabel('车道2流量')
    plt.title('车道2流量密度图')
    plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.savefig('车道2流量密度图.png')
    plt.show()

    flow3 = []
    for i, m in enumerate(flow1):

        # temp = [flow1[i] + flow2[i]]
        flow3 += [flow1[i] + flow2[i]]
    plt.scatter(density, flow3, c='b', marker='o')
    plt.xlabel('密度')
    plt.ylabel('总车道流量')
    plt.title('总车道流量密度图')
    plt.savefig('总车道流量密度图.png')
    plt.show()
    return

def main(plottype,canshu):
    '''参数设置'''

    road_length=101 #道路长度
    total_times=200 #演化长度步长
    light_time=10 #红绿灯周期长度
    rho=0.5 #整体车道密度
    p_slowdown = 0.5  # 模型里的随机刹车概率

    ##下面if种的内容不要改
    if plottype==4:
        rho = canshu  # 整体车道密度
        # light_time = 20  # 红绿灯周期长度
        # p_slowdown = 0.3  # 模型里的随机刹车概率
    elif plottype==5:
        # rho = 0.3  # 整体车道密度
        light_time = canshu  # 红绿灯周期长度
        # p_slowdown = 0.3  # 模型里的随机刹车概率

    elif plottype==3:
        # rho = 0.3  # 整体车道密度
        # light_time = 20  # 红绿灯周期长度
        p_slowdown = canshu  # 模型里的随机刹车概率
    elif plottype==6:
        rho = canshu  # 整体车道密度
        # light_time = 20  # 红绿灯周期长度
        # p_slowdown = 0.3  # 模型里的随机刹车概率
    v_max=5 #车辆最高限速

    model=0 #智能红绿灯是否开启  0为关闭  1打开
    if model==0:
        print('智能红绿灯关闭')
    else:
        print('智能红绿灯开启')
    model_traffic_light1=[]#车道1红绿灯开启记录
    model_traffic_light2 = []#道2红绿灯开启记录

    memory1 = [] # 道路1位置存储位置   里面是0  1    0 没有车    1有车
    memory2 = [] # 道路2位置存储位置   里面是0  1    0 没有车    1有车

    memory1_v = []  # 道路1位置存储速度  里面是速度值012345  如果没有车为.
    memory2_v = []  # 道路2位置存储速度  里面是速度值012345  如果没有车为.

    flow1=[] # 道路1流量存储
    flow2=[] # 道路2流量存储

    flownum1=0 # 道路1流量
    flownum2 = 0 # 道路2流量
    '''初始道路生成'''
    print('生成初始道路和车辆')
    space1, space2=Generate_road(road_length)  # 生成最初的道路
    car1, car2,space1, space2=Generate_car(road_length,rho,v_max,space1, space2)#生成最初的车辆
    # print(sum(space1),sum(space2))
    # for i,m in enumerate(space1):
    #     if m==1:
    #         plt.scatter(50,i,c='r')
    #     else:
    #         plt.scatter(50, i, c='b')
    # for i,m in enumerate(space2):
    #     if m==1:
    #         plt.scatter(i,50,c='red')
    #     else:
    #         plt.scatter(i, 50, c='black')
    # plt.show()

    '''开始演化'''
    print('开始演化')
    for time in range(total_times):
        '''判断红绿灯'''
        #判断此刻红绿灯周期  道路红绿灯的状态
        # 1如果两车道红绿灯都为0  则判断当前车道那一个车道的交叉口被占
        # 2如果设置为智能交通  则判断当前绿灯状态
        if plottype!=4 and plottype!=5 and plottype!=3 and plottype!=6:
            print('第%d轮' % time)
        # print('此时1车道有%d辆车' %sum(space1))
        # print('此时2车道有%d辆车' % sum(space2))
        space1, space2, model_traffic_light1, model_traffic_light2=Traffic_light_status(space1,space2,model,time,light_time,model_traffic_light1,model_traffic_light2,car1,car2)

        '''车辆跟随演化'''
        car1, car2, space1, space2,flownum1,flownum2=Nasch_Follow(car1, car2,space1,space2,v_max, p_slowdown,flownum1,flownum2)
        '''实时动态图'''
        # Real_time_drawing(model_traffic_light1,model_traffic_light2,space1,space2,road_length,time)
        '''统计信息'''
        memory1_temp, memory2_temp, memory1_v_temp, memory2_v_temp=statistics(car1, car2, space1, space2)
        memory1 += [memory1_temp]  # 道路1位置存储位置   里面是0  1    0 没有车    1有车
        memory2 += [memory2_temp]  # 道路2位置存储位置   里面是0  1    0 没有车    1有车

        memory1_v += [memory1_v_temp]  # 道路1位置存储速度  里面是速度值012345  如果没有车为.
        memory2_v += [memory2_v_temp]  # 道路2位置存储速度  里面是速度值012345  如果没有车为.
        ave_v1 = [x for x in memory1_v_temp if x!='.']
        ave_v2 = [x for x in memory1_v_temp if x != '.']
        # print(ave_v1)
        # print(memory1_temp)
        flow1 += [sum(memory1_temp)/len(memory1_temp)*np.mean(ave_v1)]  # 道路1流量存储
        flow2 += [sum(memory2_temp)/len(memory2_temp)*np.mean(ave_v2)]  # 道路2流量存储




    if plottype==4 or plottype==5:
        aveflow1=np.mean(flow1)
        aveflow2 = np.mean(flow2)
        if aveflow1=='Nan':
            aveflow1=0
        if aveflow2=='Nan':
            aveflow2=0
        return aveflow1,aveflow2

    if plottype==6:
        return flownum1,flownum2
    elif plottype==3:

        return memory1, memory2,total_times


    '''最后生成图'''
    ##道路时空图   横轴为cell  数轴为times
    plot1(memory1, memory2)

    ##速度的时空图
    # plot2(memory1_v, memory2_v)


    return



if __name__ == "__main__":
    '''说明'''
    ##main()这个主程序画图1时空图 图2 速度时空图  下面的 plot4()  plot5() plot3() plot6()分别画上一行注释中的图
    ##单独要某一图的时候 注释掉其他图的函数就可以
    ##单独画图1、2的时候在main()里面找到plot1() plot2() 注释掉自己不用的图就可以

    ##初始的参数设置都在main()前几行  看注释修改就可以
    # configuration-x图
    # plot3()

    ##流量current-density
    # plot4()

    # 流量与红绿灯关系图
    # plot5()

    ##流量流量密度图
    # plot6()

    #主程序
    main(0,0)