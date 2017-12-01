using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace TurtleML.Layers
{
    public class DropOutLayer : ILayer
    {
        private static readonly ThreadLocal<Random> random = new ThreadLocal<Random>(() => new Random());
        private readonly float dropOut;
        private readonly ILayer inputLayer;
        private Tensor outputs;

        private DropOutLayer(float dropOut, ILayer inputLayer)
        {
            if (dropOut < float.Epsilon)
                throw new ArgumentOutOfRangeException(nameof(dropOut), "drop out must be greater than zero.");

            this.dropOut = dropOut;
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
                Parallel.For(0, outputs.Length, h => outputs[h] = random.Value.NextDouble() >= dropOut ? inputs[h] : 0f);

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