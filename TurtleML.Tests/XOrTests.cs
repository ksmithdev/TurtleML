namespace TurtleML.Tests
{
    using System;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using TurtleML.Activations;
    using TurtleML.Initializers;
    using TurtleML.Layers;
    using TurtleML.Loss;

    [TestClass]
    public class XOrTests
    {
        [TestMethod]
        public void XOrInference()
        {
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
        }
    }
}