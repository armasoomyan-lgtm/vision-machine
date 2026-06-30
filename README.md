# vision-machine
Video-Based Deception Detection via Local-Global Facial Motion and Emotional Feature Fusion

* This project Train and Test in google.colab Envirionment.
* Dataset : RLT dataset (Pérez-Rosas et al. 2015) loads from my google drive  ('/content/gdrive')
* Storeing and Loading any files were perform at google drive
  
__ Parts & Stages of project_Code running:
 1-Extract Face from 121 video dataset
 2-Extracting packking = 15 Of all_video_np,Emotion_label_np,micro_exp_labels
 3-Extracting homograph_label_H1_np , homograph_label_mesh_np
 4- defining  model 
 5-loading inputs && outputs
 6-Pre_Testing model
 7-define epochs ,batch_size ,train_in_data train_out_data val_in_data val_out_data
 8-Training MHEmodel
 9-Logging

See the fig1 , fig2 from training responses


__ Testing & Evaluating MHEmodel:
The performance of the proposed hybrid model for deception detection was evaluated using the RLT dataset. We assessed the performance of the proposed hybrid model at two evaluation levels: (1) the GOP level, in which performance metrics are calculated using 15-frame GOPs, and (2) the video level, in which performance metrics are measured for complete test videos. 

The accuracy and AUC of the proposed hybrid architecture in GOP and video level:
Accuracy (%) --->	GOP	91.66	  Video	100

AUC          ---> GOP	0.963   Video	 1.0
