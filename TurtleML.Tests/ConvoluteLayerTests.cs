namespace TurtleML.Tests
{
    using System;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using TurtleML.Activations;
    using TurtleML.Initializers;
    using TurtleML.Layers;

    [TestClass]
    public class ConvoluteLayerTests
    {
        [TestMethod]
        public void Calculate5x5Input3x3Outputs()
        {
            var input = new StubLayer
            {
                Outputs = new Tensor(3, 3, 1)
            };

            var convolutionLayer = new ConvoluteLayer.Builder()
                .Initializer(new ConstantInitializer(1f))
                .Activation(new IdentityActivation())
                .Filters(2,2,1,1)
                .Build(input);

            var random = new Random(42);
            convolutionLayer.Initialize(random);

            var test = new Tensor(3, 3, 1);
            for(int i = 0; i < test.Length; i++)
            {
                test[i] = i;
            }

            var output = convolutionLayer.CalculateOutputs(test);

            Assert.AreEqual((2, 2, 1), output.Dimensions);
            Assert.AreEqual(9, output[0]);
            Assert.AreEqual(13, output[1]);
            Assert.AreEqual(21, output[2]);
            Assert.AreEqual(25, output[3]);
        }
    }
}
