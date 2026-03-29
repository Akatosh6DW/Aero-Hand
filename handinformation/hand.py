import socket
import serial
import numpy as np
import time
#绑定夹爪接口

class Hand_control():
    def __init__(self, port='COM3',baudrate=115200):
        self.ser = serial.Serial(port, baudrate)
        self.ser.timeout = 1  # 设置超时时间为1秒
        
        self.D_start=[0x5A] 
        #指令类型,指令长度
        self.D_mid=[0x10, 0x11]
        #例子,在每个函数里把gesture给改了就行,具体格式详见通信协议的动作控制协议
        self.gesture=[0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00,]#全张开
        #校验和
        self.xy_sum=[sum(self.D_mid+self.gesture)&0xFF]
        self.D_end=[0x5D] 
        self.gesture_serial =  self.D_start+self.D_mid+self.gesture+self.xy_sum+self.D_end 
        #self.ser.write(bytes(self.gesture_serial))
        self.HandInit()
        #力控参数组包
        self.FC=[0x40,0x28,0x00,0x01,0x64,0x4b,0x4b,0x01,0x64,0x00,0x28,0x00,0x0a,0x01,0x64,0x00,0x28,0x00,0x0a,0x01,0x64,0x00,0x5a,0x00,0x0a
                 ,0x01,0x64,0x00,0x5a,0x00,0x0a,0x01,0x64,0x00,0x5a,0x00,0x0a]#力控
        self.FCsum=[sum(self.FC)&0xFF]
        self.FCComplete=self.D_start+self.FC+self.FCsum+self.D_end
        #力控启动/停止
        self.SS=[0x4a,0x07,0x00,0x01]
        self.SSsum=[sum(self.SS)&0xFF]
        self.SSComplete=self.D_start+self.SS+self.SSsum+self.D_end
        
        #初始化手
    def HandInit(self):
        self.gesture=[0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00,]#全张开
        self.xy_sum=[sum(self.D_mid+self.gesture)&0xFF]
        self.gesture_serial =  self.D_start+self.D_mid+self.gesture+self.xy_sum+self.D_end 
        self.ser.write(bytes(self.gesture_serial))

    def HandOpen(self):
        self.gesture=[0x01, 0x75, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00,]
        self.xy_sum=[sum(self.D_mid+self.gesture)&0xFF]
        self.gesture_serial =  self.D_start+self.D_mid+self.gesture+self.xy_sum+self.D_end 
        self.ser.write(bytes(self.gesture_serial))

    def HandGrasp(self):
        self.gesture=[0x01, 0x80, 0x01, 0x40, 0x01, 0x80, 0x01, 0x80, 0x01, 0x80, 0x01, 0x80]
        self.xy_sum=[sum(self.D_mid+self.gesture)&0xFF]
        self.gesture_serial =  self.D_start+self.D_mid+self.gesture+self.xy_sum+self.D_end 
        self.ser.write(bytes(self.gesture_serial))
        
    def HandOne(self):
        self.gesture=[0x01, 0x55, 0x01, 0x55, 0x01, 0x00, 0x01, 0x55, 0x01, 0x55, 0x01, 0x55]
        self.xy_sum=[sum(self.D_mid+self.gesture)&0xFF]
        self.gesture_serial =  self.D_start+self.D_mid+self.gesture+self.xy_sum+self.D_end 
        self.ser.write(bytes(self.gesture_serial))
    def HandForceControl(self):
        self.ser.write(bytes(self.FCComplete))
    def HandForceStart(self):
        self.SS=[0x4a,0x07,0x00,0x01]
        self.SSsum=[sum(self.SS)&0xFF]
        self.SSComplete=self.D_start+self.SS+self.SSsum+self.D_end
        self.ser.write(bytes(self.SSComplete))
    def HandForceStop(self):
        self.SS=[0x4a,0x07,0x00,0x00]
        self.SSsum=[sum(self.SS)&0xFF]
        self.SSComplete=self.D_start+self.SS+self.SSsum+self.D_end
        self.ser.write(bytes(self.SSComplete))

    def Gesture_Output(self,digit):
        gesture_mapping = {
            1: self.HandInit,
            2: self.HandOpen,
            3: self.HandGrasp,
            4: self.HandOne,
            5: self.HandForceControl,
            6: self.HandForceStart,
            7: self.HandForceStop

        }
        
        if digit in gesture_mapping:
            gesture_mapping[digit]()
        else:
            print("Invalid gesture input.")



if __name__=="__main__":
    the_hand=Hand_control(port="com3")
    while 1:
        user_input = int(input("Enter a number (1-4): "))
        the_hand.Gesture_Output(user_input)
    