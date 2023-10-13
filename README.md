# Scalable-Video-Processing-Spark

[Full paper](https://drive.google.com/drive/u/0/folders/1FpsA6CeokwQgDgJ9tfLkGaYy0nDyTien)

In this work, we tackle the problem of building a scalable video processing system, where the goal is to extract,
store, and retrieve characteristic visual features from videos. We use a machine learning model, called [CLIP](https://github.com/openai/CLIP),
for supporting large set of objects and action classes and to find a list of high-precision tags for every video. To
effectively handle large video datasets, the proposed system leverages **PySpark** on top of **PyTorch** for enabling
the automatic distribution of workloads across different processes and for making fast inferences on GPU. A
**MongoDB** database server hosts the extracted set of tags for each video to support fast indexing and retrieval.

Assumptions: For this project, we assumed that our working universe consists of 1000 ImageNet object classes and 700 Kinectics action classes and the video corpus contains videos from these superset of categories. We used the pre-designed prompts from the [CLIP's repository](https://github.com/openai/CLIP/blob/main/data/prompts.md) for making zero-shot predictions on our videos to detemine their content and get a high-level information about the dataset statistics. 

# Overview
We assign the videos to different partitions on the machine and run parallel processing
<p float="left">
  <img src="https://github.com/gargsid/Scalable-Video-Processing-Spark/blob/main/figures/overview.png" width="900" height="400" />
</p> 
To reduce computation and redundancy, we do frame sampling @1 FPS before making zero-shot predictions. 
<p float="left">
  <img src="https://github.com/gargsid/Scalable-Video-Processing-Spark/blob/main/figures/frame_sample.png" width="900" height="500" />
</p> 

**We reduce the processing time as we increase the number of parallel workers or machines upto 3**. Since we only had 1 GPU, therefore in case when #workers became 4, there was high load on the GPU with 4 parallel processes.
<p float="left">
  <img src="https://github.com/gargsid/Scalable-Video-Processing-Spark/blob/main/figures/process.png" width="400" height="300" />
</p> 
Some examples for making fast query and retrieval are
<p float="left">
  <img src="https://github.com/gargsid/Scalable-Video-Processing-Spark/blob/main/figures/query.png" width="1200" height="600" />
</p> 

**Please check [notebooks/spark_inference.ipynb](https://github.com/gargsid/Scalable-Video-Processing-Spark/blob/main/notebooks/spark_inference.ipynb) for step-by-step tutorial.**

# Acknowledgement
We used [CLIP's repository](https://github.com/openai/CLIP) for getting prompts for the respective datasets and its pre-trained weights for making zero-shot predictions. 
