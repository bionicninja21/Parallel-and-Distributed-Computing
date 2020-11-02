# -- coding: utf-8 --
"""
Created on Thu Oct 31 11:56:33 2019

@author: bioni
"""

from utils import *
from matplotlib import pyplot as plt
import multiprocessing as mp
from multiprocessing import Process, Queue,Pool
import time
import glob
import threading

import subprocess
import cv2 
    
g=[0,0,0,0,0,0,0]
def f1():
    train='files/10.jpg'
    max_val = 8
    max_pt = -1
    max_kp = 0
    train_img = cv2.imread(train)
    (kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)
    good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])

    if len(good) > max_val:
        max_val = len(good)
        max_kp = kp2
    
    train1 = len(good)
    global g
    g[0]=train1
    print(train, ':' ,train1)
    

def f2():
    train='files/20.jpg'
    max_val = 8
    max_pt = -1
    max_kp = 0
    train_img = cv2.imread(train)
    (kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)
    good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])

    if len(good) > max_val:
        max_val = len(good)
        max_kp = kp2
    

    train1 = len(good)
    global g
    g[1]=train1
    print(train, ':' ,train1)    

    
    
def f3():
    train='files/50.jpg'
    max_val = 8
    max_pt = -1
    max_kp = 0
    train_img = cv2.imread(train)
    (kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)
    good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])

    if len(good) > max_val:
        max_val = len(good)
        max_kp = kp2
    
    train1 = len(good)
    global g
    g[2]=train1
    print(train, ':' ,train1)


  
def f4():
    train='files/100.jpg'
    max_val = 8
    max_pt = -1
    max_kp = 0
    train_img = cv2.imread(train)
    (kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)
    good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])

    if len(good) > max_val:
        max_val = len(good)
        max_kp = kp2
    
    train1 = len(good)
    global g
    g[3]=train1
    print(train, ':' ,train1)
    
def f5():
    train='files/200.jpg'
    max_val = 8
    max_pt = -1
    max_kp = 0
    train_img = cv2.imread(train)
    (kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)
    good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])

    if len(good) > max_val:
        max_val = len(good)
        max_kp = kp2
    
    train1 = len(good)
    global g
    g[4]=train1
    print(train, ':' ,train1)
    
def f6():
    train='files/500.jpg'
    max_val = 8
    max_pt = -1
    max_kp = 0
    train_img = cv2.imread(train)
    (kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)
    good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])

    if len(good) > max_val:
        max_val = len(good)
        max_kp = kp2
    
    train1 = len(good)
    global g
    g[5]=train1
    print(train, ':' ,train1)
    
def f7():
    train='files/2000.jpg'
    max_val = 8
    max_pt = -1
    max_kp = 0
    train_img = cv2.imread(train)
    (kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2)
    good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
    for (m, n) in all_matches:
        if m.distance < 0.789 * n.distance:
            good.append([m])

    if len(good) > max_val:
        max_val = len(good)
        max_kp = kp2
    
    train1 = len(good)
    global g
    g[6]=train1
    print(train, ':' ,train1)







max_val = 8
max_pt = -1
max_kp = 0

orb = cv2.ORB_create()

#test_img = read_img('files/test_100_2.jpg')
#test_img = read_img('files/test_2000_2.jpg')
#test_img = read_img('files/test_2000_3.jpeg')
#test_img = read_img('files/test_100_3.jpg')
test_img = read_img('files/test_20_4.jpg')

# resizing must be dynamic
# resizing must be dynamic
original = resize_img(test_img, 0.4)
display('original', original)

 
 

# keypoints and descriptors
# (kp1, des1) = orb.detectAndCompute(test_img, None)
(kp1, des1) = orb.detectAndCompute(test_img, None)

t1 = threading.Thread(target=f3) 
t2 = threading.Thread(target=f4) 
t3 = threading.Thread(target=f1) 
t4 = threading.Thread(target=f2) 
t5 = threading.Thread(target=f5) 
t6 = threading.Thread(target=f6)
t7 = threading.Thread(target=f7)    
start=time.time()

    # starting thread 1 
t1.start() 
    # starting thread 2 
t2.start() 
  

    # starting thread 1 
t3.start() 
    # starting thread 2 
t4.start() 

t5.start()
t6.start()
t7.start()
    # wait until thread 1 is completely executed 
t1.join() 
    # wait until thread 2 is completely executed 
t2.join() 

  # wait until thread 1 is completely executed 
t3.join() 
    # wait until thread 2 is completely executed 
t4.join() 

t5.join()
t6.join()
t7.join()
end=time.time()

val=g.index(max(g))
if(val==0):
    print("Detected Denomination:10")
if(val==1):
    print("Detected Denomination:20")
if(val==2):
    print("Detected Denomination:50")

if(val==3):
    print("Detected Denomination:100")

if(val==4):
    print("Detected Denomination:200")

if(val==5):
    print("Detected Denomination:500")

if(val==6):
    print("Detected Denomination:2000")    
    
    
print("Execution time:",end-start)
