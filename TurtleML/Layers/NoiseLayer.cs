using System;
using System.IO;

namespace TurtleML.Layers
{
    public class NoiseLayer : ILayer
    {
        private readonly ILayer inputLayer;
        private readonly float noise;
        private Tensor outputs;
        private Random random;

        private NoiseLayer(float noise, ILayer inputLayer)
        {
            if (noise < float.Epsilon || noise > 1f)
                throw new ArgumentOutOfRangeException(nameof(noise), "noise must be between zero and one.");

            this.noise = noise;
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
            if (training)
            {
                for (int h = 0, count = outputs.Length; h < count; h++)
                    outputs[h] = inputs[h] + ((float)random.NextDouble() * noise);

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
            private float noise = 0.1f;

            public ILayer Build(ILayer inputLayer)
            {
                return new NoiseLayer(noise, inputLayer);
            }

            public Builder Noise(float noise)
            {
                this.noise = noise;

                return this;
            }
        }
    }
}