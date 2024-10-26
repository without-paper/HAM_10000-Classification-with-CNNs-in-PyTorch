# HAM_10000-Classification-with-CNNs-in-PyTorch

Problem Statement:
Skin cancer is a major global health issue, with early detection being critical to improving treatment outcomes. However, the current manual process of diagnosing skin cancer through dermoscopy, while useful, is time-consuming, subjective, and limited by the availability of skilled dermatologists. Automated image classification methods using deep learning models have shown promise in enhancing diagnostic accuracy and efficiency, but the relationship between dataset size and model performance remains unclear. Specifically, it is unknown how different neural network architectures perform when trained on varying amounts of data, particularly in real-world scenarios where datasets may be limited or region-specific.

Project Aim:
This project aims to investigate how the size of the dataset and the complexity of neural network models influence the accuracy of skin cancer image classification. By understanding the interplay between dataset scale and model architecture, the project seeks to offer healthcare professionals better automated tools for skin cancer detection, potentially alleviating the burden on dermatologists and improving patient outcomes.

Technologies and Tools: 
- Dataset: HAM10000 dataset for training and testing [1][2]
- Python 3.8
- PyTorch
  - torchaudio-2.1.2+cu121-cp38
  - torch-2.1.2+cu121-cp38
  - torchvision-0.16.2+cu121-cp38
- Models: AlexNet, ResNet(18,34,50,101,152), ViT(Base), MLP-Mixer(B/16)

Plan and Relevance:
By systematically varying the dataset size and model complexity, this project will provide a deeper understanding of the scalability of skin cancer classification models. These findings can guide future developments in healthcare technologies, leading to more effective, efficient, and scalable diagnostic tools that reduce the dependency on specialized healthcare professionals. This project aligns with my passion for combining technology and healthcare to solve real-world problems, and I am excited to explore how AI can transform the way we diagnose and treat critical diseases like skin cancer.



[1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)", 2018; https://arxiv.org/abs/1902.03368
[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).
