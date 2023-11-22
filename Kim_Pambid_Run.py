from Object_Detector import *

#temporary
#source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
#http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
#http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz
modelurl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
#making my own tranfer network model
#modelurl=



imagepath = "E:\\Programming\\Projects\\py_tensorflow\\Transfer Learning Object Detection Project\\Test Images\\0.jpg"
#videopath = "E:\\Programming\\Projects\\py_tensorflow\\Transfer Learning Object Detection Project\\Test Images\\2.mp4"
imagemodel = "E:\\Programming\\Projects\\py_tensorflow\\Transfer Learning Object Detection Project\\Trained_Model\\checkpoints\\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\\saved_model"
classfile = "E:\\Programming\\Projects\\py_tensorflow\\Transfer Learning Object Detection Project\\Kim_Objects.names"
threshold = 0.5
detector = Object_Detector()
detector.readclasses(classfile)
detector.downloadmodel(modelurl)
detector.loadmodel(imagemodel)

detector.kim_pambid_predict_image(imagepath, threshold)
#detector.kim_pambid_predict_video(videopath,  threshold)