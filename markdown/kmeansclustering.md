## K Means Clustering

YOLOv3 uses predefined anchor boxes as reference templates for object detection. The anchors are calculated using k-means clustering. Large image datasets like MS COCO already have precomputed anchor boxes but for more specialized datasets, custom anchors may need to be computed. The following notebook is my implementation of the algorithm in python for learning purposes. 


```python
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
```


```python
data_folder_path = '../data/labels/'
```


```python
with open(Path(data_folder_path) / 'labels.txt', 'r') as f:
    lines = f.readlines()
lines
```




    ['dirt\n', 'damage']




```python
file_path = Path(f'{data_folder_path}/DJI_0004_02_05.txt')
```


```python
def get_box_data(data_folder_path):
    files = os.listdir(data_folder_path)
    boxes = []
    for i in range(len(files)):
        file_path = Path(f'{data_folder_path}/{files[i]}')
        if 'labels' not in file_path.name:
            file_boxes = np.loadtxt(file_path)
            if len(file_boxes.shape) == 1:
                file_boxes = np.expand_dims(file_boxes, axis=0)
            boxes.append(file_boxes) 
    return np.concatenate(boxes, axis = 0)
```


```python
all_boxes = get_box_data(data_folder_path)
plt.hist(all_boxes[:, 0], bins = 2)
```




    (array([ 581., 8770.]),
     array([0. , 0.5, 1. ]),
     <BarContainer object of 2 artists>)




    
![png](kmeansclustering_files/kmeansclustering_7_1.png)
    



```python
boxes = get_box_data(data_folder_path)
valid_x_y = np.logical_and(boxes[:, 1:3] >= 0, boxes[:, 1:3] <= 1)
valid_w_h = np.logical_and(boxes[:, 3:] > 0, boxes[:, 3:] <= 1)
valid_boxes = np.logical_and(valid_x_y, valid_w_h)
boxes = boxes[valid_boxes.all(axis = 1)]
```


```python
plt.hist(boxes[:, 3]/boxes[:, 4], bins = 300)
plt.xlabel('Width to Height Ratio')
plt.ylabel('Frequency')
```




    Text(0, 0.5, 'Frequency')




    
![png](kmeansclustering_files/kmeansclustering_9_1.png)
    



```python
#Handwritten IoU/Jaccard overlap function.
def IoU(boxes1, boxes2):
    """
    Calculate the Intersection over Union (IoU) of arrays of bounding boxes.
    Good to know how to do this manually instead of using a library function.

    Args:
        boxes1: (N, 4) or (4,) ndarray of float
        boxes2: (N, 4) or (4,) ndarray of float
    
    Returns:
        (N, ) ndarray of float
    
    """
    print(boxes1, boxes2)
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    
    if boxes1.ndim == 1:
        boxes1 = boxes1.reshape(1, -1)
    if boxes2.ndim == 1:
        boxes2 = boxes2.reshape(1, -1)
        
    #Convert center coordinates to top left coordinates.
    boxes1[:, :2] = boxes1[:, :2] - boxes1[:, 2:]/2
    boxes2[:, :2] = boxes2[:, :2] - boxes2[:, 2:]/2

    xA = np.maximum(boxes1[:, 0], boxes2[:, 0])
    yA = np.maximum(boxes1[:, 1], boxes2[:, 1])
    xB = np.minimum(boxes1[:, 0] + boxes1[:, 2], boxes2[:, 0] + boxes2[:, 2])
    yB = np.minimum(boxes1[:, 1] + boxes1[:, 3], boxes2[:, 1] + boxes2[:, 3])
    box_width = np.maximum(0, xB - xA)
    box_height = np.maximum(0, yB - yA)
    intersection_area = box_width * box_height
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = boxes1_area + boxes2_area - intersection_area
    iou = intersection_area / union_area
    return iou
```


```python
def IoU_width_height(boxes1, boxes2):
    """
    Calculate the Intersection over Union (IoU) of arrays of bounding boxes. 
    Assumes boxes are aligned at center. 

    Args:
        boxes1: (N, 2) or (2,) ndarray of float
        boxes2: (N, 2) or (2,) ndarray of float
    """

    if boxes1.ndim == 1:
        boxes1 = boxes1.reshape(1, -1)
    if boxes2.ndim == 1:
        boxes2 = boxes2.reshape(1, -1)
    
    intersection = np.minimum(boxes1[..., 0], boxes2[..., 0]) * np.minimum(boxes1[..., 1], boxes2[..., 1])
    union = boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    iou = intersection / union
    return iou
```

### K-Means Clustering

1. Start with K centroids by putting them in a random place. Let's use k = 9 since that's the number of anchor boxes that Yolov3 uses. I stored all of the box data in a map. 


```python
def init_centroids_rand(boxes, k):
    centroids = np.random.choice(len(boxes), k)
    box_map = {}
    for i in range(len(centroids)):
        centroid = boxes[centroids[i]]
        box_map[i] = {'center': centroid, 'boxes': []}
    return box_map

```

2. Next, assign all points to centroids based on distance metric. The YOLOv3 authors use 1 - IOU as a distance metric instead of euclidean distance.


```python

def assign_boxes(boxes, k, box_map):
    for i in range(boxes.shape[0]):
        box = boxes[i]
        iou_list = []
        for j in range(k):
            iou = 1 - IoU_width_height(box, box_map[j]['center'])
            iou_list.append(iou)
        max_iou_idx = np.argmin(iou_list)
        box_map[max_iou_idx]['boxes'].append(box)
    return box_map
```

3. Recalculate centroids by computing mean width and height of newly assigned boxes


```python
def update_centroids(box_map):
    for i in range(len(box_map)):
        if len(box_map[i]['boxes']) > 0:
            box_map[i]['center'] = np.mean(box_map[i]['boxes'], axis = 0)
```

Note: Stop the model if we've exceeded a maximum number of iterations or centroids have stopped changing.


```python
MAX_ITERATIONS = 500000
def stop_condition(old_box_map, new_box_map, iterations):
    #stop if centroids don't change
    if iterations > MAX_ITERATIONS:
        return True
    for i in range(len(new_box_map)):
        if not np.allclose(old_box_map[i]['center'], new_box_map[i]['center']):
            return False
    return True

```

Full algorithm.


```python
def k_means(boxes, k):
    old_box_map = init_centroids_rand(boxes, k)
    new_box_map = init_centroids_rand(boxes, k)
    iter = 0
    while not stop_condition(old_box_map, new_box_map, iter):
        old_box_map = new_box_map
        new_box_map = assign_boxes(boxes, k, old_box_map)
        update_centroids(new_box_map)
        iter += 1
    for key, item in new_box_map.items():
        new_box_map[key]['boxes'] = np.vstack(item['boxes'])
    return new_box_map
```


```python

def plot_clusters(boxes, box_map):
    for i, value in box_map.items():
        cluster = value["boxes"]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i}', s = 5)
    cluster_centers_ = np.array([value['center'] for value in box_map.values()])
    plt.scatter(cluster_centers_[:, 0], cluster_centers_[:, 1], color='red', marker='x', label='Centroids')
    plt.legend(loc = 'upper right')
    plt.xlabel('Width')
    plt.ylabel('Height')
```


```python
boxes_wh = boxes[:, 3:]
box_map = k_means(boxes_wh, 9)
print(box_map)
plot_clusters(boxes_wh, box_map)
```

    {0: {'center': array([0.12282679, 0.13221514]), 'boxes': array([[0.095563, 0.150943],
           [0.170302, 0.111747],
           [0.14094 , 0.148411],
           [0.088737, 0.142857],
           [0.088737, 0.118598],
           [0.088926, 0.117934],
           [0.279863, 0.121294],
           [0.090444, 0.140162],
           [0.083618, 0.134771],
           [0.168942, 0.140162],
           [0.208191, 0.140162],
           [0.093121, 0.125884],
           [0.119454, 0.121294],
           [0.174061, 0.145553],
           [0.134812, 0.150943],
           [0.133106, 0.123989],
           [0.182594, 0.148248],
           [0.114334, 0.137466],
           [0.098976, 0.123989],
           [0.068259, 0.140162],
           [0.075085, 0.12938 ],
           [0.134812, 0.145553],
           [0.192114, 0.112633],
           [0.112628, 0.115903],
           [0.090444, 0.123989],
           [0.103188, 0.124559],
           [0.087031, 0.137466],
           [0.09727 , 0.142857],
           [0.071672, 0.132075],
           [0.122867, 0.12938 ],
           [0.09215 , 0.126685],
           [0.104096, 0.137466],
           [0.141638, 0.12938 ],
           [0.146758, 0.118598],
           [0.098976, 0.148248],
           [0.085324, 0.132075],
           [0.087031, 0.12938 ],
           [0.06314 , 0.140162],
           [0.240614, 0.140162],
           [0.049488, 0.140162],
           [0.073379, 0.137466],
           [0.199659, 0.137466],
           [0.076792, 0.12938 ],
           [0.221843, 0.150943],
           [0.356655, 0.132075],
           [0.151877, 0.110512],
           [0.167235, 0.121294],
           [0.104096, 0.142857],
           [0.09727 , 0.113208],
           [0.105705, 0.127209],
           [0.093121, 0.117934],
           [0.170648, 0.118598],
           [0.1843  , 0.107817],
           [0.16041 , 0.148248],
           [0.090604, 0.127209],
           [0.046075, 0.142857],
           [0.120805, 0.109983],
           [0.127986, 0.140162],
           [0.056314, 0.148248],
           [0.087031, 0.137466],
           [0.179181, 0.126685],
           [0.100683, 0.148248],
           [0.069966, 0.132075],
           [0.109215, 0.126685],
           [0.066553, 0.134771],
           [0.117747, 0.132075],
           [0.223549, 0.142857],
           [0.167235, 0.121294],
           [0.197987, 0.131185],
           [0.09727 , 0.142857],
           [0.162116, 0.115903],
           [0.15529 , 0.126685],
           [0.088737, 0.148248],
           [0.237201, 0.150943],
           [0.081911, 0.115903],
           [0.112628, 0.134771],
           [0.186007, 0.113208],
           [0.119454, 0.118598],
           [0.05802 , 0.148248],
           [0.117747, 0.145553],
           [0.208191, 0.137466],
           [0.09215 , 0.132075],
           [0.059727, 0.145553],
           [0.095563, 0.113208],
           [0.138225, 0.132075],
           [0.083618, 0.145553],
           [0.114334, 0.145553],
           [0.093857, 0.123989],
           [0.172355, 0.115903],
           [0.237201, 0.115903],
           [0.087031, 0.142857],
           [0.075085, 0.134771],
           [0.102389, 0.132075],
           [0.118289, 0.13781 ],
           [0.095563, 0.145553],
           [0.100683, 0.140162],
           [0.03413 , 0.148248],
           [0.093857, 0.148248],
           [0.119454, 0.140162],
           [0.087031, 0.123989],
           [0.146758, 0.123989],
           [0.156997, 0.115903],
           [0.358362, 0.121294],
           [0.146758, 0.107817],
           [0.110738, 0.14046 ],
           [0.122867, 0.115903],
           [0.107509, 0.113208],
           [0.119454, 0.126685],
           [0.163591, 0.115722],
           [0.100671, 0.128534],
           [0.37884 , 0.137466],
           [0.153584, 0.148248],
           [0.12116 , 0.150943],
           [0.093857, 0.142857],
           [0.078498, 0.123989],
           [0.078498, 0.137466],
           [0.088737, 0.134771],
           [0.206485, 0.140162],
           [0.190436, 0.111308],
           [0.109215, 0.150943],
           [0.151877, 0.134771],
           [0.068259, 0.137466],
           [0.081911, 0.115903],
           [0.087031, 0.12938 ],
           [0.080205, 0.132075],
           [0.087031, 0.134771],
           [0.196246, 0.126685],
           [0.09215 , 0.115903],
           [0.119454, 0.145553],
           [0.088737, 0.137466],
           [0.095563, 0.121294],
           [0.067114, 0.13516 ],
           [0.098976, 0.148248],
           [0.119454, 0.150943],
           [0.146758, 0.115903],
           [0.109899, 0.117047],
           [0.201365, 0.123989],
           [0.080205, 0.118598],
           [0.12116 , 0.134771],
           [0.109215, 0.145553],
           [0.087031, 0.118598],
           [0.149329, 0.126323],
           [0.081911, 0.118598],
           [0.180887, 0.132075],
           [0.35151 , 0.074644],
           [0.098976, 0.126685],
           [0.109215, 0.140162],
           [0.107383, 0.119259],
           [0.110922, 0.123989],
           [0.093857, 0.142857],
           [0.136519, 0.137466],
           [0.088737, 0.126685],
           [0.088737, 0.132075],
           [0.09727 , 0.12938 ],
           [0.116041, 0.134771],
           [0.187713, 0.123989],
           [0.12116 , 0.126685],
           [0.107509, 0.134771],
           [0.093857, 0.140162],
           [0.162116, 0.132075],
           [0.150171, 0.121294],
           [0.105705, 0.128534],
           [0.119966, 0.13781 ],
           [0.168942, 0.132075],
           [0.24744 , 0.148248],
           [0.094799, 0.127209],
           [0.109215, 0.142857],
           [0.110922, 0.148248],
           [0.186007, 0.150943],
           [0.240614, 0.132075],
           [0.12116 , 0.132075],
           [0.088737, 0.142857],
           [0.127986, 0.123989],
           [0.104096, 0.115903],
           [0.085324, 0.140162],
           [0.069966, 0.142857],
           [0.09215 , 0.132075],
           [0.098993, 0.119259],
           [0.108221, 0.14046 ],
           [0.073379, 0.140162],
           [0.095563, 0.123989],
           [0.105802, 0.126685],
           [0.153584, 0.123989],
           [0.083618, 0.137466],
           [0.119454, 0.118598],
           [0.075085, 0.134771],
           [0.136519, 0.134771],
           [0.153584, 0.12938 ],
           [0.146758, 0.12938 ],
           [0.322526, 0.12938 ],
           [0.172355, 0.113208],
           [0.085324, 0.132075],
           [0.133106, 0.110512],
           [0.286689, 0.148248],
           [0.136519, 0.113208],
           [0.088737, 0.140162],
           [0.16041 , 0.115903],
           [0.079698, 0.13781 ],
           [0.14849 , 0.148411],
           [0.09727 , 0.118598],
           [0.146758, 0.140162],
           [0.129693, 0.12938 ],
           [0.116611, 0.148411],
           [0.128356, 0.14311 ],
           [0.114334, 0.12938 ],
           [0.088737, 0.121294],
           [0.071672, 0.140162],
           [0.080205, 0.123989],
           [0.126678, 0.149736],
           [0.087031, 0.137466],
           [0.040956, 0.142857],
           [0.098976, 0.118598],
           [0.056314, 0.148248],
           [0.145051, 0.107817],
           [0.088737, 0.142857],
           [0.127986, 0.148248],
           [0.120805, 0.148411],
           [0.092282, 0.120584],
           [0.039249, 0.148248],
           [0.095563, 0.145553],
           [0.116041, 0.132075],
           [0.100683, 0.115903],
           [0.098976, 0.121294],
           [0.148464, 0.118598],
           [0.069966, 0.140162],
           [0.09727 , 0.134771],
           [0.049488, 0.137466],
           [0.180369, 0.108658],
           [0.197952, 0.132075],
           [0.076792, 0.123989],
           [0.162116, 0.118598],
           [0.071672, 0.142857],
           [0.109215, 0.115903],
           [0.085324, 0.121294],
           [0.208191, 0.134771],
           [0.152685, 0.118372],
           [0.051195, 0.140162],
           [0.12628 , 0.134771],
           [0.083618, 0.148248],
           [0.085324, 0.132075],
           [0.088737, 0.115903],
           [0.039249, 0.145553],
           [0.197952, 0.145553],
           [0.081911, 0.145553],
           [0.110922, 0.134771],
           [0.119454, 0.113208],
           [0.12628 , 0.145553],
           [0.090604, 0.128534],
           [0.087248, 0.116608],
           [0.165529, 0.113208],
           [0.075085, 0.126685],
           [0.090444, 0.140162],
           [0.082215, 0.117934],
           [0.143456, 0.151061],
           [0.119966, 0.127209],
           [0.085324, 0.148248],
           [0.087031, 0.134771],
           [0.090444, 0.145553],
           [0.150171, 0.126685],
           [0.180887, 0.121294],
           [0.104096, 0.145553],
           [0.052901, 0.137466],
           [0.080205, 0.12938 ],
           [0.140101, 0.113072],
           [0.12116 , 0.115903],
           [0.088737, 0.115903],
           [0.255973, 0.118598],
           [0.116041, 0.150943],
           [0.203072, 0.140162],
           [0.134812, 0.121294],
           [0.131399, 0.148248],
           [0.226962, 0.126685],
           [0.165529, 0.140162],
           [0.117747, 0.126685],
           [0.192833, 0.150943],
           [0.080537, 0.128534],
           [0.139932, 0.145553],
           [0.203072, 0.137466],
           [0.073379, 0.142857],
           [0.15529 , 0.132075],
           [0.117747, 0.110512],
           [0.16041 , 0.105121],
           [0.150168, 0.151061],
           [0.182594, 0.105121],
           [0.124573, 0.115903],
           [0.119454, 0.150943],
           [0.138225, 0.12938 ],
           [0.100683, 0.137466],
           [0.09727 , 0.137466],
           [0.174061, 0.126685],
           [0.150171, 0.121294],
           [0.046075, 0.140162],
           [0.215017, 0.134771],
           [0.079698, 0.13516 ],
           [0.026846, 0.149736],
           [0.096477, 0.13781 ],
           [0.069966, 0.132075],
           [0.117747, 0.148248],
           [0.095563, 0.145553],
           [0.061433, 0.148248],
           [0.068792, 0.13251 ],
           [0.088737, 0.140162],
           [0.118289, 0.14046 ],
           [0.191126, 0.137466],
           [0.09727 , 0.123989],
           [0.238908, 0.105121],
           [0.103188, 0.147086],
           [0.082215, 0.148411],
           [0.093857, 0.113208],
           [0.040956, 0.142857],
           [0.076792, 0.145553],
           [0.085324, 0.115903],
           [0.126678, 0.13516 ],
           [0.058725, 0.144435],
           [0.131399, 0.145553],
           [0.12116 , 0.115903],
           [0.167235, 0.115903],
           [0.047782, 0.142857],
           [0.165529, 0.113208],
           [0.05802 , 0.145553],
           [0.071672, 0.140162],
           [0.134812, 0.145553],
           [0.090444, 0.145553],
           [0.331058, 0.145553],
           [0.129693, 0.12938 ],
           [0.087031, 0.142857],
           [0.075085, 0.137466],
           [0.044369, 0.140162],
           [0.100683, 0.150943],
           [0.100683, 0.142857],
           [0.069966, 0.145553],
           [0.083618, 0.118598],
           [0.166107, 0.123234],
           [0.087031, 0.118598],
           [0.230375, 0.142857],
           [0.174061, 0.142857],
           [0.102389, 0.150943],
           [0.278157, 0.150943],
           [0.170648, 0.115903],
           [0.100683, 0.123989],
           [0.09727 , 0.12938 ],
           [0.09727 , 0.134771],
           [0.228669, 0.140162],
           [0.131399, 0.110512],
           [0.116041, 0.126685],
           [0.156997, 0.148248],
           [0.100683, 0.148248],
           [0.081911, 0.134771],
           [0.134812, 0.12938 ],
           [0.129693, 0.12938 ],
           [0.076792, 0.148248],
           [0.163823, 0.137466],
           [0.21843 , 0.137466],
           [0.150171, 0.134771],
           [0.049488, 0.142857],
           [0.213311, 0.150943],
           [0.083618, 0.113208],
           [0.131399, 0.123989],
           [0.106544, 0.14311 ],
           [0.204778, 0.137466],
           [0.165529, 0.110512],
           [0.312287, 0.140162],
           [0.069966, 0.134771],
           [0.071672, 0.137466],
           [0.179181, 0.118598],
           [0.090444, 0.145553],
           [0.215017, 0.137466],
           [0.090444, 0.150943],
           [0.170648, 0.140162],
           [0.105802, 0.150943],
           [0.440273, 0.088949],
           [0.288396, 0.107817],
           [0.078498, 0.121294],
           [0.09215 , 0.115903],
           [0.174061, 0.140162],
           [0.031879, 0.145761],
           [0.076342, 0.124998],
           [0.114334, 0.110512],
           [0.07802 , 0.127209],
           [0.165529, 0.118598],
           [0.083618, 0.12938 ],
           [0.105802, 0.145553],
           [0.105802, 0.132075],
           [0.127986, 0.126685],
           [0.186007, 0.134771],
           [0.05802 , 0.140162],
           [0.081911, 0.12938 ],
           [0.148464, 0.110512],
           [0.068259, 0.145553],
           [0.098976, 0.12938 ],
           [0.162116, 0.110512],
           [0.136519, 0.121294],
           [0.088737, 0.123989],
           [0.100683, 0.132075],
           [0.099832, 0.147086],
           [0.102389, 0.123989],
           [0.049488, 0.142857],
           [0.080205, 0.118598],
           [0.030717, 0.148248],
           [0.087031, 0.148248],
           [0.259228, 0.147524],
           [0.09396 , 0.13781 ],
           [0.099832, 0.127209],
           [0.082215, 0.128973],
           [0.150171, 0.140162],
           [0.090444, 0.148248],
           [0.068259, 0.134771],
           [0.046075, 0.142857],
           [0.083618, 0.123989],
           [0.098976, 0.140162],
           [0.061433, 0.140162],
           [0.085324, 0.150943],
           [0.186242, 0.116608],
           [0.068792, 0.133835],
           [0.143345, 0.137466],
           [0.083618, 0.123989],
           [0.114334, 0.113208],
           [0.075085, 0.137466],
           [0.095563, 0.134771],
           [0.240614, 0.115903],
           [0.156997, 0.137466],
           [0.145051, 0.142857],
           [0.059727, 0.134771],
           [0.127986, 0.126685],
           [0.100683, 0.150943],
           [0.075085, 0.134771],
           [0.087031, 0.12938 ],
           [0.334471, 0.110512],
           [0.127986, 0.121294],
           [0.201365, 0.137466],
           [0.095563, 0.118598],
           [0.046075, 0.142857],
           [0.104096, 0.142857],
           [0.078498, 0.126685],
           [0.197952, 0.113208],
           [0.044369, 0.145553],
           [0.081911, 0.132075],
           [0.100683, 0.118598],
           [0.095638, 0.117934],
           [0.124573, 0.142857],
           [0.061242, 0.14311 ],
           [0.071672, 0.132075],
           [0.073379, 0.132075],
           [0.151877, 0.118598],
           [0.151877, 0.105121],
           [0.180887, 0.142857],
           [0.075085, 0.132075]])}, 1: {'center': array([0.07063511, 0.08463375]), 'boxes': array([[0.03413 , 0.097035],
           [0.117747, 0.086253],
           [0.115772, 0.087456],
           ...,
           [0.03413 , 0.086253],
           [0.042662, 0.075472],
           [0.052901, 0.061995]])}, 2: {'center': array([0.11368751, 0.22391323]), 'boxes': array([[0.022184, 0.277628],
           [0.196246, 0.167116],
           [0.07047 , 0.195228],
           ...,
           [0.066553, 0.215633],
           [0.141638, 0.371968],
           [0.046075, 0.16442 ]])}, 3: {'center': array([0.04134635, 0.03458901]), 'boxes': array([[0.031879, 0.027827],
           [0.051195, 0.037736],
           [0.075085, 0.024259],
           ...,
           [0.051195, 0.043127],
           [0.035836, 0.045822],
           [0.02901 , 0.043127]])}, 4: {'center': array([0.02272601, 0.02693048]), 'boxes': array([[0.020478, 0.024259],
           [0.020478, 0.02965 ],
           [0.027304, 0.026954],
           ...,
           [0.020973, 0.019876],
           [0.025168, 0.025177],
           [0.027304, 0.018868]])}, 5: {'center': array([0.00985122, 0.02004211]), 'boxes': array([[0.011945, 0.013477],
           [0.010239, 0.016173],
           [0.005034, 0.029152],
           ...,
           [0.006711, 0.017226],
           [0.008389, 0.023852],
           [0.009228, 0.021202]])}, 6: {'center': array([0.012902  , 0.03869923]), 'boxes': array([[0.011945, 0.026954],
           [0.011945, 0.02965 ],
           [0.015358, 0.02965 ],
           ...,
           [0.010906, 0.088781],
           [0.006711, 0.047703],
           [0.009228, 0.050354]])}, 7: {'center': array([0.02677691, 0.06583859]), 'boxes': array([[0.018771, 0.102426],
           [0.035235, 0.048142],
           [0.025597, 0.045822],
           ...,
           [0.030717, 0.078167],
           [0.022184, 0.06469 ],
           [0.018771, 0.048518]])}, 8: {'center': array([0.52652757, 0.62906728]), 'boxes': array([[1.      , 0.423181],
           [0.317114, 0.999561],
           [0.635067, 0.174913],
           ...,
           [1.      , 0.336927],
           [0.411263, 0.994609],
           [0.477816, 1.      ]])}}



    
![png](kmeansclustering_files/kmeansclustering_24_1.png)
    


Random initialization often leads to suboptimal centroids that may be clustered too close together. Notice how our dataset has two distinct groups of boxes with much larger widths and heights. We want the centroid with the largest width and height (the purple star) to be even closer to those groups. Hence, we use k means plus plus. 

### K-Means Plus Plus

1. Randomly select the first centroid from the data points.
2. For each data point compute its distance from the nearest, previously chosen centroid.
3. Select the next centroid from the data points such that the probability of choosing a point as centroid is directly proportional to its distance from the nearest, previously chosen centroid. (i.e. the point having maximum distance from the nearest centroid is most likely to be selected next as a centroid)
4. Repeat steps 2 and 3 until k centroids have been sampled


```python
def k_means_plus_plus(boxes, k):
    box_map = {}
    centroids = []
    #pick a random centroid
    centroid = boxes[np.random.choice(len(boxes))]
    centroids.append(centroid)
    for i in range(1, k):
        #Calculate distances from nearest centroid
        distances = []
        for j in range(boxes.shape[0]):
            # Find the closest centroid for each box
            min_distance = min(1 - IoU_width_height(centroid, boxes[j]) for centroid in centroids)
            distances.append(min_distance)
        #Choose the next centroid based on squared distances
        squared_distances = (np.array(distances) ** 2).flatten()
        probabilities = squared_distances/np.sum(squared_distances)
        next_idx = np.random.choice(len(boxes), p=probabilities)
        centroids.append(boxes[next_idx])
    for i in range(len(centroids)):
        box_map[i] = {'center': centroids[i], 'boxes': []}
    return box_map
```


```python
better_box_map = k_means_plus_plus(boxes_wh, 9)
```


```python
for i, value in better_box_map.items():
    plt.scatter(value['center'][0], value['center'][1], s = 100, marker = '*', edgecolors= 'black')
plt.xlabel('Width')
plt.ylabel('Height')
```




    Text(0, 0.5, 'Height')




    
![png](kmeansclustering_files/kmeansclustering_30_1.png)
    


##### Final algorithm using k means plus plus to initialize centroids


```python
def k_means_improved(boxes, k):
    old_box_map = init_centroids_rand(boxes, k)
    new_box_map = k_means_plus_plus(boxes, k)
    iter = 0
    while not stop_condition(old_box_map, new_box_map, iter):
        old_box_map = new_box_map
        new_box_map = assign_boxes(boxes, k, old_box_map)
        update_centroids(new_box_map)
        iter += 1
    for key, item in new_box_map.items():
        new_box_map[key]['boxes'] = np.vstack(item['boxes'])
    return new_box_map
```


```python
box_map = k_means_improved(boxes_wh, 9)
#Sort centroids by size
sorted_centroids = sorted(box_map.items(), key = lambda x: x[1]['center'][0] * x[1]['center'][1], reverse = True)
for key, value in sorted_centroids:
    print(f'Centroid {key}: {value["center"]}')
```

    Centroid 8: [0.87446633 0.53306323]
    Centroid 4: [0.31980473 0.76646783]
    Centroid 3: [0.42159424 0.27818934]
    Centroid 0: [0.15216333 0.19993068]
    Centroid 7: [0.20414104 0.07815664]
    Centroid 5: [0.06523085 0.19887119]
    Centroid 2: [0.09718028 0.09445283]
    Centroid 1: [0.03726812 0.07139271]
    Centroid 6: [0.01791475 0.02998719]



```python
plot_clusters(boxes_wh, box_map)
```


    
![png](kmeansclustering_files/kmeansclustering_34_0.png)
    


Now you can see that we've settled on more optimal anchors that are farther spread out and thus encompass the various widths and heights of the data's boxes. Let's use SciKit Learn's K-means clustering implementation to see how our implementation fares. 


```python
def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two boxes.
    Boxes are in (width, height) format.
    """
    intersection = np.minimum(box1[0], box2[0]) * np.minimum(box1[1], box2[1])
    union = (box1[0] * box1[1]) + (box2[0] * box2[1]) - intersection
    return intersection / union
```


```python
def custom_distance(box1, box2):
    """
    Custom distance metric based on 1 - IoU.
    """
    return 1 - iou(box1, box2)
```


```python
from sklearn.cluster import KMeans

class CustomKMeans:
    def __init__(self, n_clusters, max_iter=1000, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None

    def fit(self, X):
        # Initialize centroids
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=1, n_init=1)
        kmeans.fit(X)
        self.cluster_centers_ = kmeans.cluster_centers_

        for _ in range(self.max_iter):
            # Assign clusters based on custom distance
            distances = np.array([[custom_distance(x, center) for center in self.cluster_centers_] for x in X])
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.all(np.abs(new_centers - self.cluster_centers_) < self.tol):
                break

            self.cluster_centers_ = new_centers

    def predict(self, X):
        distances = np.array([[custom_distance(x, center) for center in self.cluster_centers_] for x in X])
        return np.argmin(distances, axis=1)
```


```python
# Example usage
boxes_wh = boxes[:, 3:]

n_clusters = 9
custom_kmeans = CustomKMeans(n_clusters=n_clusters)
custom_kmeans.fit(boxes_wh)

# Plot clusters
import matplotlib.pyplot as plt

labels = custom_kmeans.predict(boxes_wh)
sorted_centroids = np.array(sorted(custom_kmeans.cluster_centers_, key = lambda x: x[0] * x[1], reverse = True))
print("Sorted clusters")
print(sorted_centroids)
for i in range(n_clusters):
    cluster = boxes_wh[labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i}', s = 5)
plt.scatter(custom_kmeans.cluster_centers_[:, 0], custom_kmeans.cluster_centers_[:, 1], color='red', marker='x', label='Centroids')
plt.xlabel('Width')
plt.ylabel('Height')
plt.legend(loc = 'upper right')
plt.title('KMeans Clustering')
plt.show()
```

    Sorted clusters
    [[0.43708957 0.95413507]
     [0.96614968 0.34158547]
     [0.21688574 0.47689602]
     [0.26286556 0.19207433]
     [0.11327912 0.18252201]
     [0.10629132 0.07504841]
     [0.05065195 0.12488623]
     [0.03883405 0.05335556]
     [0.01562711 0.03437581]]



    
![png](kmeansclustering_files/kmeansclustering_39_1.png)
    


So it looks like our algorithm could use some improvement. But it's a good start! This notebook was meant to be more of a learning exercise. 
