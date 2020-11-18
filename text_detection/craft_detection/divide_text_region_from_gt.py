import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

import craft_utils
import imgproc


# bboxes, polys, score_text = test_net(net, image, args.text_threshold, 
# args.link_threshold, args.low_text, args.cuda, args.poly)
if __name__ == '__main__':
    # 参数设置
    canvas_size = 1280
    mag_ratio = 1.0 # 图像放大倍数
    text_threshold = 0.7 # region_map阈值
    low_text = 0.4 # text low-bound score
    link_threshold = 0.001 # affinity_map阈值
    poly = False # 是否输出多边形框，默认输出四个点的框
    
    
    root = './data'
    img_list = os.listdir(os.path.join(root,'img'))
    
    for img_path in img_list:
        image_path = os.path.join(root, 'img', img_path)
        affinity_path = os.path.join(root, 'affinity', img_path.split('_')[0] + '_affinity_' \
                    + img_path.split('_')[1].split('.')[0] + '.npy')
        region_path = os.path.join(root, 'region', img_path.split('_')[0] + '_region_' \
                    + img_path.split('_')[1].split('.')[0] + '.npy')
        anno_path = os.path.join(root, 'anno', img_path.split('.')[0] + '.json')
        
        image = np.array(plt.imread(image_path)) # 225*517*3
        region = np.load(region_path)
        affinity = np.load(affinity_path)
        
        # resize
        # img_resized=352*800*3, target_ratio=1.5
        # size_heatmap=400*176, ratio_h=w=0.66666667
#        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
#        ratio_h = ratio_w = 1 / target_ratio
#        plt.imshow(img_resized.astype(np.int))
#        region = cv2.resize(region,(img_resized.shape[1]//2,img_resized.shape[0]//2))
#        affinity = cv2.resize(affinity,(img_resized.shape[1]//2,img_resized.shape[0]//2))
        
        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(region, affinity, text_threshold, link_threshold, low_text, poly)
        
        # coordinate adjustment
#        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
#        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
#        for k in range(len(polys)):
#            if polys[k] is None: 
#                polys[k] = boxes[k]
        
        # render results (optional)
        render_img = region.copy()
        render_img = np.hstack((render_img, affinity))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)
        for i, box in enumerate(boxes):
            _,(kernel_w,kernel_h),_ = cv2.minAreaRect(box) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            kernel_w,kernel_h = int(kernel_w),int(kernel_h)
            if kernel_w < kernel_h:
                kernel_w,kernel_h = kernel_h,kernel_w
            
            box = np.array(box).astype(np.int32).reshape((-1))
            box = box.reshape(-1, 2)
#            cv2.polylines(image, [box.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            # 将高斯核透视变换，坐标(列，行)[box.reshape((-1, 1, 2))]
            src = np.float32(box) # 左上，左下，右下，右上
            tgt = np.float32([(0,0),(kernel_w,0),(kernel_w,kernel_h),(0,kernel_h)])
            M = cv2.getPerspectiveTransform(src, tgt)
            dst = cv2.warpPerspective(image, M, (kernel_w,kernel_h)) # dst就是所要的文本图像
        
        '''读取标注文件中的字符'''
        f=open(anno_path,encoding='utf-8')
        anno = json.load(f)
        shapes = anno['shapes']
        
        text = []
        for s in shapes:
            text.append(s['label'])
        text = ''.join(text)
        
        save_path = './divides'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        print(text)
        cv2.imwrite(os.path.join(save_path,text+'.jpg'),dst)
        
#            cv2.imshow('win',dst)
#            if cv2.waitKey() == 0xFF & ord('q'):
#                cv2.destroyAllWindows()
#                import sys
#                sys.exit()
#            break

#        break