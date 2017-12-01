using System;
using System.Diagnostics;
using System.IO;

namespace TurtleML.Layers
{
    public class DropOutLayer : ILayer
    {
        //private static readonly ThreadLocal<Random> random = new ThreadLocal<Random>(() => new Random());
        private readonly float dropOut;

        private readonly ILayer inputLayer;
        private readonly Random random;
        private Tensor outputs;

        private DropOutLayer(float dropOut, Random random, ILayer inputLayer)
        {
            if (dropOut < float.Epsilon)
                throw new ArgumentOutOfRangeException(nameof(dropOut), "drop out must be greater than zero.");

            this.dropOut = dropOut;
            this.random = random;
            this.inputLayer = inputLayer ?? throw new ArgumentNullException(nameof(inputLayer));

            var inputs = inputLayer.Outputs;
            outputs = new Tensor(inputs.Dimensions);
        }

        public Tensor Outputs => outputs;

        public void Backpropagate(Tensor errors, float learningRate)
        {
            inputLayer.Backpropagate(errors, learningRate);
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            Debug.Assert(inputs.Length == outputs.Length, $"Your input array (size: {inputs.Length}) does not match the specified size of {outputs.Length}.");

            if (training)
            {
                for (int h = 0, count = outputs.Length; h < count; h++)
                    outputs[h] = random.NextDouble() >= dropOut ? inputs[h] : 0f;

                return outputs;
            }

            return inputs;
        }

        public void Dump(BinaryWriter writer)
        {
        }

        public void Restore(BinaryReader reader)
        {
        }

        public class Builder : ILayerBuilder
        {
            private float dropOut;
            private Random random;

            public ILayer Build(ILayer inputLayer)
            {
                return new DropOutLayer(dropOut, random, inputLayer);
            }

            public Builder DropOut(float dropOut)
            {
                this.dropOut = dropOut;

                return this;
            }

            public Builder Seed(Random random)
            {
                this.random = random;

                return this;
            }
        }
    }
}