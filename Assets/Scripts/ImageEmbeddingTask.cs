using UnityEngine;
using Unity.Sentis;
using UnityEngine.Rendering;
using System.Threading.Tasks;
using UnityEngine.TextCore.LowLevel;
using TMPro;
using UnityEngine.UI;
using System.Collections.Generic;

public class ImageEmbeddingTask : MonoBehaviour
{
    public Texture2D inputTexture1;
    public Texture2D inputTexture2;

    public RawImage inputImage1;
    public RawImage inputImage2;
    public TMP_Text resultText;

    public ModelType modelType = ModelType.mobilenet_v3_small;

    ModelAsset embeddingModelAsset;
    Model embeddingModel;
    Worker embeddingWorker1;
    Worker embeddingWorker2;
    float[] results;

    BackendType backendType = BackendType.GPUCompute;

    Worker dotProductWorker;
    Model dotProductModel;

    public enum ModelType
    {
        mobilenet_v3_small,
        mobilenet_v3_large,
        dinov2_vits14,
        swav_imagenet_layer2_sim,
        convnext_base_384_22k_1k
    }

    List<TensorShape> tensorShapes = new List<TensorShape>
    {
        new TensorShape(1, 224, 224, 3),
        new TensorShape(1, 224, 224, 3),
        new TensorShape(1, 3, 224, 224),
        new TensorShape(1, 3, 224, 224),
        new TensorShape(1, 3, 224, 224)
    };

    List<int> embeddingSizes = new List<int>
    {
        1024,
        1280,
        384,
        1024,
        1000
    };

    List<string> modelNames = new List<string>
    {
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "dinov2_vits14",
        "swav_imagenet_layer2_sim",
        "convnext_base_384_22k_1k"
    };

    void Start()
    {
        embeddingModelAsset = Resources.Load<ModelAsset>($"Models/{modelNames[(int)modelType]}");

        //Image embedding model
        embeddingModel = ModelLoader.Load(embeddingModelAsset);
        embeddingWorker1 = new Worker(embeddingModel, backendType);
        embeddingWorker2 = new Worker(embeddingModel, backendType);

        //Dot Product model
        FunctionalGraph dotProductGraph = new FunctionalGraph();
        FunctionalTensor input1 = dotProductGraph.AddInput(DataType.Float, new TensorShape(1, embeddingSizes[(int)modelType]));
        FunctionalTensor input2 = dotProductGraph.AddInput(DataType.Float, new TensorShape(1, embeddingSizes[(int)modelType]));
        FunctionalTensor input1Mag = Functional.Sqrt(Functional.ReduceSumSquare(input1, -1));
        FunctionalTensor input2Mag = Functional.Sqrt(Functional.ReduceSumSquare(input2, -1));
        FunctionalTensor dotProduct = Functional.ReduceSum(Functional.Mul(input1, input2), -1) / (input1Mag * input2Mag);
        dotProductModel = dotProductGraph.Compile(dotProduct);
        dotProductWorker = new Worker(dotProductModel, backendType);
    }

    void Update()
    {
        inputImage1.texture = inputTexture1;
        inputImage2.texture = inputTexture2;

        using Tensor inputTensor1 = TextureConverter.ToTensor(inputTexture1, width: 224, height: 224, channels: 3);
        inputTensor1.Reshape(tensorShapes[(int)modelType]);
        embeddingWorker1.Schedule(inputTensor1);
        using Tensor<float> outputTensor1 = embeddingWorker1.PeekOutput() as Tensor<float>;

        using Tensor inputTensor2 = TextureConverter.ToTensor(inputTexture2, width: 224, height: 224, channels: 3);
        inputTensor2.Reshape(tensorShapes[(int)modelType]);
        embeddingWorker2.Schedule(inputTensor2);
        using Tensor<float> outputTensor2 = embeddingWorker2.PeekOutput() as Tensor<float>;

        outputTensor1.Reshape(new TensorShape(1, embeddingSizes[(int)modelType]));
        outputTensor2.Reshape(new TensorShape(1, embeddingSizes[(int)modelType]));

        dotProductWorker.SetInput(0, outputTensor1);
        dotProductWorker.SetInput(1, outputTensor2);
        dotProductWorker.Schedule();

        using Tensor<float> outputTensor = dotProductWorker.PeekOutput() as Tensor<float>;
        results = outputTensor.DownloadToArray();
        //Debug.Log($"Dot product result: {results[0]}");
        resultText.text = $"{modelNames[(int)modelType]}: {results[0]}";
    }


    void OnDisable()
    {
        embeddingWorker1.Dispose();
        embeddingWorker2.Dispose();
        dotProductWorker.Dispose();
    }
}
