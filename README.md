# [Sensing and Automation in Agri-System (SAAS) Lab](https://sites.google.com/view/xin-zhang-lab/home)
## [Department of Agricultural & Biological Engineering](https://www.abe.msstate.edu/)
## [Mississippi State University](https://www.msstate.edu/)

# Why do we create this repo?
*This repo is created for accompanying the conference publication of **"Automating catfish cutting process using deep learning-based semantic segmentation"** in Sensing for Agriculture and Food Quality and Safety XV (Vol. 12545, pp. 103-116). SPIE. [DOI](https://doi.org/10.1117/12.2663370)*

***Abstract:** Mississippi and Alabama are the top two states producing and processing catfish in the United States, with the annual production of $382 million in 2022. The catfish industry supplies protein-rich catfish products to the U.S. market and contributes considerably to the development of the local economy. However, the traditional catfish processing heavily relies on human labors leading to a high demand of workforce in the processing facilities. De-heading, gutting, portioning, filleting, skinning, and trimming are the main steps of the catfish processing, which normally require blade-based cutting device (e.g., metal blades) to handle. The blade-based manual catfish processing might lead to product contamination, considerable fish meat waste, and low yield of catfish fillet depending on the workers’ skill levels. Furthermore, operating the cutting devices may expose the human labors to undesired work accidents. Therefore, automated catfish cutting process appears to be an alternative and promising solution with minimal involvement of human labors. To further enable, assist, and automate the catfish cutting technique in near real-time, this study presents a novel computer vision-based sensing system for segmenting the catfish into different target parts using deep learning and semantic segmentation. In this study, 396 raw and augmented catfish images were used to train, validate, and test five state-of-the-art deep learning semantic segmentation models, including BEiTV1, SegFormer-B0, SegFormer-B5, ViT-Adapter and PSPNet. Five classes were pre-defined for the segmentation, which could effectively guide the cutting system to locate the target, including the head, body, fins, tail of the catfish, and the image background. Overall, BEiTV1 demonstrated the poorest performance with 77.3% of mIoU (mean intersection-over-union) and 86.7% of MPA (mean pixel accuracy) among all tested models using the test data set, while SegFormer-B5 outperformed all others with 89.2% of mIoU and 94.6% of MPA on the catfish images. The inference speed for SegFormer-B5 was 0.278 sec per image at the resolution of 640x640. The proposed deep learning-based sensing system is expected to be a reliable tool for automating the catfish cutting process.*

# Who owns the data shared in this repo?
*This repo is owned by the [Sensing and Automation in Agri-System (SAAS) Lab](https://sites.google.com/view/xin-zhang-lab/home) in the Department of Agricultural & Biological Engineering at Mississippi State University. Please contact the lab PI, [Dr. Xin Zhang](https://www.abe.msstate.edu/people/faculty/xin-zhang/), if you have any questions regarding this repo.*

# How to use this repo?
- [**Data**](https://github.com/Zhanglab-abe/CatFish-Segment/tree/main/Data)
  - Catfish images acquired in a processing plant in Alabama, USA (JPG format)
  - Annotations of catfish images created manually (PNG format)

- [**Split_and_Augmentations**](https://github.com/Zhanglab-abe/CatFish-Segment/tree/main/Split_and_Augmentations)
  - Developed Python code (IPYNB) for splitting the dataset, exporting the image masks from the LabelBox, and augmenting the images for a larger dataset.

- [**Pixel_wise_analysis**](https://github.com/Zhanglab-abe/CatFish-Segment/tree/main/Pixel_wise_analysis)
  - Developed Python code (PY) for checking the pre-defined five (5) segmentation classes in the dataset, including:
    - `Background`
    - `Head`
    - `Body`
    - `Fins`
    - `Tails`

- [**Models**](https://github.com/Zhanglab-abe/CatFish-Segment/tree/main/Models)
  - Deployed deep learning-based semantic segmentation models in this work, including:
    - BEiT
    - PSPNet
    - SegFormer
      - B0
      - B5
    - ViT-Adapter

- [**Trained_weights**](https://github.com/Zhanglab-abe/CatFish-Segment/tree/main/Trained_weights)
  - BEiT (2.1 GB)
  - [PSPNet](https://github.com/Zhanglab-abe/CatFish-Segment/blob/main/Trained_weights/PSPNet.pth) (520 MB)
  - SegFormer
    - [B0](https://github.com/Zhanglab-abe/CatFish-Segment/tree/main/Trained_weights/SegFormerB0/checkpoint-7200/optimizer.pt) (28.5 MB)
    - [B5](https://github.com/Zhanglab-abe/CatFish-Segment/tree/main/Trained_weights/SegFormerB5/optimizer.pt) (646 MB)
  - ViT-Adapter (5.5 GB)

- [**Results**](https://github.com/Zhanglab-abe/CatFish-Segment/tree/main/Results)
  - Test inference of catfish segmentation using the trained models
  - Typical failures in catfish segmentation due to the small size of the imagery dataset

# How to properly cite us if you find this repo useful?
*To cite this repo in your works, use the following BibTeX entry:*

```bibtex
@inproceedings{thayananthan2023automating,
  title={Automating catfish cutting process using deep learning-based semantic segmentation},
  author={Thayananthan, Thevathayarajh and Zhang, Xin and Liu, Wenbo and Yao, Tianqi and Huang, Yanbo and Wijewardane, Nuwan K and Lu, Yuzhen},
  booktitle={Sensing for Agriculture and Food Quality and Safety XV},
  volume={12545},
  pages={103--116},
  year={2023},
  organization={SPIE}
}
```
