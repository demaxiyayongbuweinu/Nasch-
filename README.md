# Nasch-封闭十字路口的交通流模型 nagel-schreckenberg model，模拟车流，计算平均车流量和平均速度

一位学生花钱让写的代码，正在研究相关问题，被骗后分享给大家，不旺自己花这么多时间

模型规则如下：
普通nasch跟随模型
![)61F0)%OQKE`CIS5FOP{ 6R](https://user-images.githubusercontent.com/89890506/131606013-35a93196-ff00-4c95-aae3-b0aca86866f0.jpg)

红绿灯三种模式：1.关闭（就是没有）  2.常规红绿灯（一红一绿）3.智能红绿灯（检测密度，给水平主干道多个绿灯，比如三绿一红）
智能交通灯的意思其实就是监测密度，然后决定给水平还是竖直开红绿灯，比如说给水平开了一次绿灯之后。我发现水平道路的这个密度还是比竖直道路的密度要大。这个时候我就继续给水平道路开一次绿灯，但是要设置一个上限，比如说给某一条道路连续开三次绿灯之后，就必须给这条道路红灯

输出的东西：

1.两条道路的kymograph，有车的cell为黑色1×1方格，没车的就是空白，横轴为车道cell，竖轴为模拟的times

2.速度密度图

3.对于三种红绿灯模式，画水平和竖直两个车道的流量-密度关系图

4.同样的流量J计算方法，想画的关系图是flow-T
这里的T是指红绿灯的周期，也就是light_time这个参数，light_time=20的意思就是红绿灯每20秒为一个周期，想画的图是flow和T的关系图， 比如T=3， T=10， T=30这三种情况下，应该输出三个图

![YMFBHR}GP9OE`X`_}06TG{I](https://user-images.githubusercontent.com/89890506/131606150-b51278fd-a90a-496c-b160-302ea4a84bb8.jpg)

5.configuration-x关系图
