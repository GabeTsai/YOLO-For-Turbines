# YOLO-For-Turbines
Replicating YOLO v3 architecture according to the paper using PyTorch. Using pretrained yolov3.weights file, the model can detect and classify objects in images from the MSMSCOCO dataset. Currently fine tuning the model to detect dust and damage on wind turbines.

**Link to project:** 

## How It's Made:

**Tech used:** Python, Streamlit

**Model architecture**: You-Only-Look-Once v3, or YOLOv3. Built from scratch following tutorial by @[Aladdin Persson](https://github.com/aladdinpersson) using PyTorch according to the official paper from @[Joseph Redmon](https://github.com/pjreddie). Added custom functions to load in just the trained backbone or fully trained model from the official YOLOv3 repo. 

**Custom Data Pipeline**: 
- Built a custom PyTorch Dataset to load in images and annotations and augment them according to model architecture
- Dataset supports multi scale training - every 10 batches, images in batches are enlarged at random by n*32 increment, following the official training procedure in the paper. 
- Wrote functions to convert yolo-formatted predictions, which are tensors at three different scales with format (batch_size, num_anchors = 3, grid_size, grid_size, bbox_coords + object_score + class_predictions), to bounding boxes for plotting on original image.

**Mosaic Augmentation**:
- Taking a page out of the training for YOLOv4, implemented a custom function that performs mosaic augmentation: [Mosaic Augmentation](examples/MosaicAugmentation.png)
- Loads four images in a 2x2 mosaic, then selects a point from the top left corner of the image that is between 20 to 30 percent of the entire mosaic.
- Then, select a cutout from the chosen corner and adjust the boxes in that cutout to match the cutout. 
- This feature took quite a while to implement because there were a lot of edge cases to consider, such as making sure that one quadrant of the mosaic contained objects, and accounting for the fact that not every cutout would create a mosaic with objects. 

## Optimizations
**Vectorizing Non-Max Supression**:
- In order to calculate mAP or get the model's final predictions, we have to consider that the model will produce a lot of box predictions for the same object. Using Non-Max Suppression, we select boxes with the highest confidence score and remove all other boxes that have an iou greater than or equal to that of a threshold we designate. Rinse and repeat until you get the filtered, final model box predictions.
- I noticed during model training that 
- I changed an inner for loop to use vectorized PyTorch code instead, resulting in an over 10x speed boost
## Lessons Learned:
- Implementing models is a hard but insightful learning experience that forces you to really understand the model architecture/nuances
- You augment images to help the network see the image in a new but applicable context for your problem. For example, if your task is detecting small defects on wind turbines, a blur augmentation will harm model training because it will be even harder to distinguish the defect from the image. You would want augmentations that preserve or enhance the visbility of these defects.

