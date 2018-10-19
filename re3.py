# -*- coding: UTF-8 -*-
import caffe 
import cv2
import numpy as np

#将图片resize模型输入尺寸227x227
def img_resize(img1,img2):
    out_img1 = cv2.resize(img1,(227,227),interpolation=cv2.INTER_AREA)
    out_img2 = cv2.resize(img2,(227,227),interpolation=cv2.INTER_AREA)
    return out_img1, out_img2

#局部搜索区域
def corp_ss(frame, tc):
    x1 = int(tc[0]-0.5*(tc[2]-tc[0]))
    x2 = int(tc[2]+0.5*(tc[2]-tc[0]))
    y2 = int(tc[3]+0.5*(tc[3]-tc[1]))
    y1 = int(tc[1]-0.5*(tc[3]-tc[1]))
    roi = frame[y1:y2,x1:x2]
    return roi,[x1,y1]

#加载模型
def load_model(prototxt, caffemodel):
    caffe.set_mode_gpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    return net

#图片预处理通道转置并减去均值
def img_preprocessing(img1,img2,net):
    transformer = caffe.io.Transformer({'image_data': net.blobs['image_data'].data.shape})
    img_mean = np.array([123.151630838, 115.902882574, 103.062623801], dtype=np.float32)
    transformer.set_transpose('image_data', (2, 0, 1))
    transformer.set_mean('image_data', img_mean)
    out_img1 = transformer.preprocess('image_data', img1)
    out_img2 = transformer.preprocess('image_data', img2)
    return out_img1, out_img2

#局部坐标系转全局坐标
def coord_trans(roi ,bbox,track_bbox):
    th, tw, _ = roi.shape
    scale_w = float(tw)/227. 
    scale_h = float(th)/227.
    x1,x2 = bbox[0]*scale_w+track_bbox[0],bbox[2]*scale_w+track_bbox[0]
    y1,y2 = bbox[1]*scale_h+track_bbox[1],bbox[3]*scale_h+track_bbox[1]
    return x1,y1,x2,y2


#前向运算
def forward(net,img1,img2,lstmdata):
    #开辟内存
    net.blobs['image_data'].reshape(2, 3, 227, 227)
    net.blobs['init_loop_output1'].reshape(1, 1024, 1, 1)
    net.blobs['init_loop_cell1'].reshape(1, 1024, 1, 1)
    net.blobs['init_loop_output2'].reshape(1, 1024, 1, 1)
    net.blobs['init_loop_cell2'].reshape(1, 1024, 1, 1)

    #输入数据
    net.blobs['image_data'].data[0] =img2
    net.blobs['image_data'].data[1] =img1
    net.blobs['init_loop_output1'].data[...] =lstmdata[0].reshape((1,1024,1,1))
    net.blobs['init_loop_cell1'].data[...] =lstmdata[1].reshape((1,1024,1,1))
    net.blobs['init_loop_output2'].data[...] =lstmdata[2].reshape((1,1024,1,1))
    net.blobs['init_loop_cell2'].data[...] =lstmdata[3].reshape((1,1024,1,1))
    out = net.forward()
    lstmdata[0] = net.blobs['loop_output1_00'].data[0]
    lstmdata[1] = net.blobs['loop_cell1_00'].data[0]
    lstmdata[2] = net.blobs['loop_output2_00'].data[0]
    lstmdata[3] = net.blobs['loop_cell2_00'].data[0]
    ####################
    return (out["output_xyxy"][0]/10)*227


prototxt = "RNNNet_deploy.prototxt"
caffemodel = "deploy_weights.caffemodel"
#初始化加载模型
net = load_model(prototxt, caffemodel)
cap = cv2.VideoCapture("1.avi")
#初始坐标框
track_bbox = [246, 226, 340, 340]
#lstm初始化输入向量
lstmdata = [np.zeros((1,1024)) for _ in range(4)]
#判断是否初始化
INIT = 0

while True:
    ret, frame = cap.read()
    df = frame.copy()
    #局部搜索roi
    roi, rc = corp_ss(frame, track_bbox)
    if INIT ==  0:
        previous_roi = roi
        INIT=1
    #图像resize 227x227
    img1, img2 = img_resize(roi,previous_roi)
    #图像预处理
    input_img1, input_img2 = img_preprocessing(img1,img2,net)
    #前向预测
    bbox = forward(net,input_img1, input_img2,lstmdata)
    #坐标转换
    x1,y1,x2,y2 = coord_trans(roi ,bbox,rc)
    #更新跟踪框
    track_bbox = [int(x1), int(y1), int(x2), int(y2)]
    #更新t-1时刻roi
    previous_roi = roi
    cv2.rectangle(df, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.imshow("df", df)
    #cv2.imshow("img1", img1)
    #cv2.imshow("img2", img2)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
