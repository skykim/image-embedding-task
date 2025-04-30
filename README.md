# image-embedding-task
Calculate cosine similarity for various image embedding models including: MobileNet-V3 (small/large), DinoV2, Swav, ConvNeXt

## Requirements ##
- Unity 6000.1.0f1
- Sentis 2.1.2

## Models (onnx) ##
- MobileNet-V3 (small/large) [(Link)](https://ai.google.dev/edge/mediapipe/solutions/vision/image_embedder)
- dinov2-small-onnx [(Link)](https://huggingface.co/sefaburak/dinov2-small-onnx)
- imagenet-swav-resnet50w2 [(Link)](https://huggingface.co/lixiangchun/imagenet-swav-resnet50w2)
- convnext-base-384-22k-1k [(Link)](https://huggingface.co/Xenova/convnext-base-384-22k-1k)

## Setup ##

### 1. Download Unity Project ###
- Download Models [(Models.zip)](https://drive.google.com/file/d/1dEd56Puu_vNS7onPHmz3plYh-6nwZQQH/view?usp=drive_link) 
- Unzip to /Assets/Resources/Models/

### 2. Run a scene ###
- Run /Assets/Scenes/ImageEmbeddingScene.unity


<img width="512" alt="image" src="https://github.com/user-attachments/assets/7dec1ccb-94da-4724-b9d4-80e4ac4ffda5" />
<img width="512" alt="Image" src="https://github.com/user-attachments/assets/fb2172ee-9b4d-4492-aea4-685dbd4fa77f" />
