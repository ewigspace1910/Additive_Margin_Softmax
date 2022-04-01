# Additive_Margin_Softmax

##Table of Content
 1. **Softmax Loss** *[done]*
 2. **Cosloss** *[done]*
 3. **Arcloss** *[done]*
 4. **Li-Arcloss** *[done]*
 5. Sphereloss *[todo]*
 6. Magloss *[todo]*

## Data Prepare

The official InsightFace project open their training data in the [DataZoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). 

The directory should have a structure like this:

```
read_dir/
  - id1/
    -- id1_1.jpg
    ...
  - id2/
    -- id2_1.jpg
    ...
  - id3/
    -- id3_1.jpg
    -- id3_2.jpg
    ...
  ...
```

##Training Tips(Continual updates)

During the competition of LFR, we found some useful training tricks for face recognition.

* We tried training model from scratch with ArcFace, but diverged after some epoch. Since the embedding size is smaller than 512. If u want try with smaller embeds(128, 64,...), u should use Pre trained model then finetune before train it. Airface(in the references) is another solution.
* In 512-dimensional embedding feature space, it is difficult for the lightweight model to learn the distribution of the features.I think resnet50 is a standard.
* If u can't use large batch size(>128), you should use small learning rate.
* If ur system not strong, rescale your dataset (maybe 100id, 100imgs/id), small batch size, then adjust hyper-parameter (s ~ 5-10)
* the optimal setting for m of ArcFace is between 0.45 and 0.5.

##References

  - [Addtive margin for softmax](https://arxiv.org/pdf/1801.05599.pdf)
  - [AirFace:Lightweight and Efficient Model](https://arxiv.org/pdf/1907.12256.pdf)
  - [ArcFace: Additive Angular Margin Loss](https://arxiv.org/pdf/1801.07698.pdf)
  - [CosFace: Large Margin Cosine Loss](https://arxiv.org/pdf/1801.09414.pdf)
  - [SphereFace: Deep Hypersphere Embedding](https://arxiv.org/pdf/1704.08063.pdf)
  - [MagFace: A Universal Representation](https://arxiv.org/pdf/2103.06627.pdf)
