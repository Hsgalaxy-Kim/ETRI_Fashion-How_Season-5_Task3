# ETRI_Fashion-How_Season-5_Task3
This is the repository for ETRI_Fashion-How Season 5 task 3 by VCL.

## 1. Goal

The goal of this task is to develop a model that ranks three outfit candidates based on user conversations. In this process, the model sequentially learns a total of six tasks following an incremental learning setting, where it is crucial to retain knowledge from previous tasks without forgetting.

Keyword: `Task-Free Incremental Learning`

## 2. Conditions

* Augmentation : Yes
* Ensemble : No
* Memory : No
* Multimodal : Yes
* External Data : No
* External Model : Yes
* Minimum model performance: 0.7  
* Use of previous model features: No

## 3. Metric

WKT(Weighted Kenall's Tau)

example
| Real rank | Pred. rank | WKT    |
|----------|--------------|--------|
| 2,1,0    | 2,1,0        | 1.0000 |
| 2,1,0    | 2,0,1        | 0.5455 |
| 2,1,0    | 1,2,0        | 0.1818 |
| 2,1,0    | 1,0,2        | -0.3636 |
| 2,1,0    | 0,2,1        | -0.3636 |
| 2,1,0    | 0,1,2        | -1.0000 |

## 4. Model structure

![teaser](https://github.com/user-attachments/assets/4a8f1210-5065-48c4-b630-be8e784b35cd)

## 5. Setup

* Python : 3.8.19
* CUDA : 11.3
* Library : `requirements.txt`

## Training with validation

  ```
  bash run_example.sh
  ```

## Validation Scores

| Task | Task1_val | Task2_val | Task3_val | Task4_val | Task5_val | Task6_val | Mean   |
|------|-----------|-----------|-----------|-----------|-----------|-----------|--------|
| 1    | 0.8348    |           |           |           |           |           | 0.8348 |
| 2    | 0.7455    | 0.7632    |           |           |           |           | 0.7544 |
| 3    | 0.8064    | 0.6097    | 0.7768    |           |           |           | 0.7310 |
| 4    | 0.8210    | 0.6912    | 0.5578    | 0.8133    |           |           | 0.7208 |
| 5    | 0.7809    | 0.6996    | 0.4921    | 0.7255    | 0.7434    |           | 0.6883 |
| 6    | 0.8131    | 0.6377    | 0.5774    | 0.7299    | 0.6906    | 0.7261    | 0.6958 |
    
## Final Score
(after quantization)

0.7063361

Sever: 16.8MB

My PC: 17.6 MB
