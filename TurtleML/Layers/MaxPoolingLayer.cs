using System;
using System.IO;

namespace TurtleML.Layers
{
    public sealed class MaxPoolingLayer : ILayer
    {
        private readonly int sampleHeight;
        private readonly int sampleWidth;
        private readonly Tensor signals;
        private readonly (int x, int y, int z)[,,] switches;

        private MaxPoolingLayer(int sampleWidth, int sampleHeight, IOutput input)
        {
            if (sampleWidth < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(sampleWidth), "sample width must be greater than zero");
            }
            if (sampleHeight < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(sampleHeight), "sample height must be greater than zero");
            }
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            this.sampleWidth = sampleWidth;
            this.sampleHeight = sampleHeight;

            var inputs = input.Outputs;
            (int inputWidth, int inputHeight, int inputDepth) = inputs.Dimensions;

            int outputWidth = (inputWidth + (inputWidth % sampleWidth)) / sampleWidth;
            int outputHeight = (inputHeight + (inputHeight % sampleHeight)) / sampleHeight;

            Outputs = new Tensor(outputWidth, outputHeight, inputDepth);
            signals = new Tensor(inputWidth, inputHeight, inputDepth);
            switches = new (int, int, int)[outputWidth, outputHeight, inputDepth];
        }

        public Tensor Outputs { get; }

        public Tensor Backpropagate(Tensor errors, float learningRate, float momentumRate)
        {
            signals.Clear();

            for (int z = 0; z < Outputs.Depth; z++)
            {
                for (int y = 0; y < Outputs.Height; y++)
                {
                    for (int x = 0; x < Outputs.Width; x++)
                    {
                        (int sX, int sY, int sZ) = switches[x, y, z];

                        signals[sX, sY, sZ] = errors[x, y, z];
                    }
                }
            }

            return signals;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            Outputs.Clear(float.MinValue);

            for (int z = 0; z < inputs.Depth; z++)
            {
                for (int y = 0; y < inputs.Height; y++)
                {
                    for (int x = 0; x < inputs.Width; x++)
                    {
                        int px = x / sampleWidth;
                        int py = y / sampleHeight;

                        if (Outputs[px, py, z] < inputs[x, y, z])
                        {
                            Outputs[px, py, z] = inputs[x, y, z];
                            switches[px, py, z] = (x, y, z);
                        }
                    }
                }
            }

            return Outputs;
        }

        public void Dump(BinaryWriter writer)
        {
        }

        public void Initialize(Random random)
        {
        }

        public void Restore(BinaryReader reader)
        {
        }

        public class Builder : ILayerBuilder
        {
            private int sampleHeight;
            private int sampleWidth;

            public ILayer Build(IOutput input)
            {
                return new MaxPoolingLayer(sampleWidth, sampleHeight, input);
            }

            public Builder Sample(int width, int height)
            {
                sampleWidth = width;
                sampleHeight = height;

                return this;
            }
        }
    }
}