#guide url: https://www.youtube.com/watch?v=2yQqg_mXuPQ
#make a way to detect gpu
#microsoft vitual c++ Redistributable
#pip install tensorflow-gpu==2.6
#pip opencv-python
#command promp: tf.test.is_gpu_available
#tf.config.list_physical_devices('GPU')

#cv2 sa file directory
import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

def create_bounding_box(self, image):
    inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
    inputTensor = inputTensor[tf.newaxis,...]

    detections = delf.model(inputTensor)

    bboxs = detections["detection_boxes"][0].numpy()
    classIndexes = detections["detection_classes"][0].numpy().astype(np.int32)
    classScore = detections["detection_scores"][0].numpy()

    imH, imW, imC = image.shape #image height, weight , and width

    bboxidx = tf.image.non_max_supression(bboxs, classScores, max_output_size=50,
    iou_threshold=0.5, score_threshold=0.5)

    if len(bboxidx) !=0:
      for i in range(0, len(bboxidx)):
        bbox = tuple(bboxs[i].tolist())
        classConfidence = round(100*classScores[i])
        classIndex = classIndexes[i]

        classLabelText = self.classesList[classIndex]
        classColor = self.colorList[classIndex]

        displayText = '{}: {}%'.format(classLabelText, classConfidence)

        ymin, xmin, ymax, xmax = bbox

        xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

        cv2.rectangle(image,(xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
        cv2.putText(image, displayText, (xmin, ymin -10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
        ##Border Top####################################################################
        lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))
        ##Left##
        cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
        cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)
        ##Right##
        cv2.line(image, (xmax, ymin), (xmax + lineWidth, ymin), classColor, thickness=5)
        cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)
        ##Border End#####################################################################
        ##Left##
        cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
        cv2.line(image, (xmin, ymax), (xmin, ymax + lineWidth), classColor, thickness=5)
        ##Right##
        cv2.line(image, (xmax, ymax), (xmax + lineWidth, ymax), classColor, thickness=5)
        cv2.line(image, (xmax, ymax), (xmax, ymax + lineWidth), classColor, thickness=5)

  return image



