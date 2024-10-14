# Multisensory Glove
This work proposes a low-cost multisensory glove
(≈ 140 USD) equipped with commercial piezoresistive force
sensors (FSRs) and inertial measurement units (IMUs) for object
recognition. A set of 28 daily life objects is used to evaluate the
glove by applying grasping actions. The raw signals acquired
through these actions are employed directly to train a shallow
one-dimensional convolutional neural network. The network is
deployed on a low-cost edge system to perform online object
recognition, thereby presenting an end-to-end system. The proposed system achieves a classification accuracy of 99.29% while
consuming only 59.85 mW and 0.4875 mJ of power and energy
per inference respectively.

# Daily life objects
The following image represents a photograph of the 28 daily life objects used in this experiment.
<img src="https://github.com/user-attachments/assets/7e95b9d1-6913-4d6f-ae8b-628f5724284c" alt="Objects" width="300">



# End-to-End System
![System_page-0001](https://github.com/user-attachments/assets/0df7af4f-dfc2-4542-af8f-8336d07f52db)
The figure present the end-to-end system for object recognition: (a) Front and back view of the glove developed for the experiment. (b) Data acquisition board and the
LabView GUI for data collection. (c) An example of collected data, followed by the architecture of the 1D-CNN used in this work for identifying objects and
the board used for performing real-time inference.
