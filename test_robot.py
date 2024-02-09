#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import machinevisiontoolbox as mvtb
import matplotlib.pyplot as plt
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot

from model import *

CUDA_AVALIABLE = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
parser.add_argument('--model', type=str, default='', help="Filename of model to be used")
args = parser.parse_args()

bot = PiBot(ip=args.ip)

# stop the robot 
bot.setVelocity(0, 0)

#INITIALISE NETWORK HERE
# nn_model = AlexNet()


#LOAD NETWORK WEIGHTS HERE
base_path = "/Users/eric/Documents/phd/RVSS_Need4Speed/"
model_path = base_path + "self-driving-robot/models/" + args.model
nn_model = torch.load(model_path, map_location="cpu")


if CUDA_AVALIABLE:
    nn_model.cuda()

nn_model.eval()

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

stop_blocker = 0
stopped_last = False

idx = 0

angles_list = [0.0]

try:
    angle = 0
    while True:
        start = time.time()
        
        # get an image from the the robot
        img_array = bot.getImage() # numpy ndarray

        # crop image
        img_array = img_array[np.ceil(img_array.shape[0]/2).astype(int):, :]

        cv2.imshow('image', img_array)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            bot.setVelocity(0, 0)
            break

        

        # convert to rgb
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        

        #TO DO: apply any necessary image transforms
        

        #TO DO: pass image through network get a prediction
        # print(img_array.shape)
        # plt.imshow(img_array)
        # plt.show()
        img = torch.tensor(img_array).float().unsqueeze(0).permute(0, 3, 1, 2)
        
        
        # img = img - img.mean(axis=(1, 2, 3), keepdims=True)
        
        
        if CUDA_AVALIABLE:
            img.cuda()
        
        # angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # angle = angles[idx]
        # idx += 1
        # start = time.time()
        
        idx += 1
        angle = nn_model(img)# torch.tensor(angles_list[idx-1]))
        angles_list.append(angle)
        # print(f"inf time: {time.time() - start}")

        print(f"angle: {angle.item()}")
        
        
        #TO DO: convert prediction into a meaningful steering angle
        Kd = 20 #base wheel speeds, increase to go faster, decrease to go slower
        # if abs(angle) < 0.1:
            # Kd = 40
        Ka = 40 #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)

        # #TO DO: check for stop signs?
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        img_array = cv2.blur(img_array, (5, 5))
                
        red = img_array * np.expand_dims(img_array[:, :, 0] >= 170, 2)
        red = red * np.expand_dims(180 >= img_array[:, :, 0], 2)
        red = red * np.expand_dims(img_array[:, :, 1] >= 130, 2)
        
        pixels = np.sum(red != 0)
        
        stop_blocker -= 1
        
        # pixels 500
        if pixels >= 800: #stop_blocker <= 0:
            if not stopped_last:
                # stop for some time
                print("\n\n\n================ STOPPED ===============\n\n\n")
                stopped_last = True
                bot.setVelocity(0, 0)
                time.sleep(5)
        else:
            stopped_last = False
            stop_blocker = 40
            
        bot.setVelocity(left, right)
        print(time.time() - start)
        # time.sleep(0.08)
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
