# TurtleML README

**TurtleML** is a **toy machine learning library** implemented in C#, primarily designed for educational purposes. It is *not* intended for production use due to its lack of performance optimizations, limited feature set, and simplistic design.

## Overview

This project implements basic neural network components such as:
- Various activation functions (e.g., ReLU, Sigmoid, Tanh)
- Different layer types (Fully Connected, Convolutional, Dropout, Reshape, etc.)
- Initialization strategies (He, Xavier, Constant, Random Uniform, Zero)
- Loss functions (Mean Squared Error, Cross Entropy, Mean Absolute Error)
- Learning policies (Fixed, Step Decay, Time Decay)

The library includes a basic `TrainingNetwork` and an `InferenceNetwork`, allowing for training and testing of neural networks.

## Example: XOR Problem

Here's the XOR example from `XOrTests.cs`. This test demonstrates how to create and train a simple neural network to solve the XOR problem:

```csharp
var seed = new Random(42);
var network = new TrainingNetwork.Builder()
    .Loss(new MeanSquareError())
    .Seed(seed)
    .Layers(
        new ReshapeLayer.Builder().Dimensions(2),
        new FullyConnectedLayer.Builder()
            .Outputs(3)
            .Initializer(new HeInitializer(), new ZeroInitializer())
            .Activation(new LeakyReLUActivation()),
        new FullyConnectedLayer.Builder()
            .Outputs(1)
            .Initializer(new HeInitializer(), new ZeroInitializer())
            .Activation(new SigmoidActivation())
    )
    .Build();

var trainingSet = new TensorSet
{
    { Tensor.Create([0f, 0f]), Tensor.Create([0f]) },
    { Tensor.Create([1f, 0f]), Tensor.Create([1f]) },
    { Tensor.Create([0f, 1f]), Tensor.Create([1f]) },
    { Tensor.Create([1f, 1f]), Tensor.Create([0f]) }
};

float finalError = 0f;
network.TrainingProgress += (s, e) => finalError = e.TrainingError;
network.Fit(trainingSet, trainingSet, 2_000);

var output1 = network.CalculateOutputs(Tensor.Create([0f, 0f]))[0];
var output2 = network.CalculateOutputs(Tensor.Create([1f, 0f]))[0];
var output3 = network.CalculateOutputs(Tensor.Create([0f, 1f]))[0];
var output4 = network.CalculateOutputs(Tensor.Create([1f, 1f]))[0];

Assert.AreEqual(0.0, Math.Round(output1, 0));
Assert.AreEqual(1.0, Math.Round(output2, 0));
Assert.AreEqual(1.0, Math.Round(output3, 0));
Assert.AreEqual(0.0, Math.Round(output4, 0));
```

This example:
- Creates a network with two fully connected layers.
- Uses LeakyReLU and Sigmoid activation functions.
- Trains the network on the XOR dataset for 2000 epochs.
- Verifies that the trained network correctly approximates the XOR function.

## Limitations

- **Performance**: The library is not optimized for speed or memory efficiency.
- **Feature Set**: It lacks many features found in production ML libraries (e.g., GPU support, advanced optimizers).
- **Robustness**: No error handling or validation beyond basic assumptions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you'd like to improve this library, feel free to submit a pull request. However, please keep in mind that it's intended as an educational tool and not a production-grade framework.

---

**Note:** This is a simplified README for the TurtleML project. For more detailed information about specific components or methods, refer to the source code documentation.
