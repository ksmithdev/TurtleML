using System;
using System.Diagnostics;
using System.IO;

namespace TurtleML.Layers
{
    public class DropOutLayer : ILayer
    {
        private readonly float dropOut;
        private readonly ILayer inputLayer;
        private Tensor outputs;
        private Random random;

        private DropOutLayer(float dropOut, ILayer inputLayer)
        {
            if (dropOut < float.Epsilon || dropOut > 1f)
                throw new ArgumentOutOfRangeException(nameof(dropOut), "drop out must be between zero and one.");

            this.dropOut = dropOut;
            this.inputLayer = inputLayer ?? throw new ArgumentNullException(nameof(inputLayer));

            var inputs = inputLayer.Outputs;
            outputs = new Tensor(inputs.Dimensions);
        }

        public Tensor Outputs => outputs;

        public Tensor Backpropagate(Tensor errors, float learningRate, float momentumRate)
        {
            return errors;
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

        public void Initialize(Random random)
        {
            this.random = random ?? new Random();
        }

        public void Restore(BinaryReader reader)
        {
        }

        public class Builder : ILayerBuilder
        {
            private float dropOut;

            public ILayer Build(ILayer inputLayer)
            {
                return new DropOutLayer(dropOut, inputLayer);
            }

            public Builder DropOut(float dropOut)
            {
                this.dropOut = dropOut;

                return this;
            }
        }
    }
}