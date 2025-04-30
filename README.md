# image-embedding-task
A sample code for calculating and comparing cosine similarity between images using various image embedding models in Unity.

## Requirements ##
- Unity 6000.1.0f1
- Sentis 2.1.2

## Models (onnx) ##
- **MobileNet-V3 Small/Large**: Optimized for efficient mobile-based embeddings [(Link)](https://ai.google.dev/edge/mediapipe/solutions/vision/image_embedder)
  - ```python -m tf2onnx.convert --opset 15 --tflite mobilenet_v3_large.tflite --output mobilenet_v3_large.onnx```
- **DinoV2**: Self-supervised learning-based high-performance image representation [(Link)](https://huggingface.co/sefaburak/dinov2-small-onnx)
- **Swav**: Clustering-based self-supervised learning embeddings [(Link)](https://huggingface.co/lixiangchun/imagenet-swav-resnet50w2)
- **ConvNeXt**: Modern convolutional architecture-based embeddings [(Link)](https://huggingface.co/Xenova/convnext-base-384-22k-1k)

## Setup ##

### 1. Download Unity Project ###
- Download image embedding models [(Models.zip)](https://drive.google.com/file/d/1dEd56Puu_vNS7onPHmz3plYh-6nwZQQH/view?usp=drive_link) 
- Unzip to /Assets/Resources/Models/

### 2. Run a scene ###
- Run /Assets/Scenes/ImageEmbeddingScene.unity


## Screenshot ##

<img width="768" alt="image" src="https://github.com/user-attachments/assets/7dec1ccb-94da-4724-b9d4-80e4ac4ffda5" />

<img width="768" alt="Image" src="https://github.com/user-attachments/assets/fb2172ee-9b4d-4492-aea4-685dbd4fa77f" />
