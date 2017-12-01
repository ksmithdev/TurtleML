using System;
using System.IO;

namespace TurtleML.Layers
{
    public class MaxPoolingLayer : ILayer
    {
        private readonly ILayer inputLayer;
        private readonly Tensor outputs;
        private readonly int sampleHeight;
        private readonly int sampleWidth;
        private readonly Tensor signals;
        private readonly (int x, int y, int z)[,,] switches;

        private MaxPoolingLayer(int sampleWidth, int sampleHeight, ILayer inputLayer)
        {
            if (sampleWidth < 1)
                throw new ArgumentOutOfRangeException(nameof(sampleWidth), "sample width must be greater than zero");
            if (sampleHeight < 1)
                throw new ArgumentOutOfRangeException(nameof(sampleHeight), "sample height must be greater than zero");

            this.sampleWidth = sampleWidth;
            this.sampleHeight = sampleHeight;
            this.inputLayer = inputLayer ?? throw new ArgumentNullException(nameof(inputLayer));

            var inputs = inputLayer.Outputs;
            (int inputWidth, int inputHeight, int inputDepth) = inputs.Dimensions;

            int outputWidth = inputWidth / sampleWidth;
            int outputHeight = inputHeight / sampleHeight;

            outputs = new Tensor(outputWidth, outputHeight, inputDepth);
            signals = new Tensor(inputWidth, inputHeight, inputDepth);
            switches = new(int, int, int)[outputWidth, outputHeight, inputDepth];
        }

        public Tensor Outputs => outputs;

        public void Backpropagate(Tensor errors, float learningRate)
        {
            signals.Clear();

            for (int z = 0; z < outputs.Depth; z++)
                for (int y = 0; y < outputs.Height; y++)
                    for (int x = 0; x < outputs.Width; x++)
                    {
                        (int sX, int sY, int sZ) = switches[x, y, z];

                        signals[sX, sY, sZ] = errors[x, y, z];
                    }

            inputLayer.Backpropagate(signals, learningRate);
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            outputs.Clear();

            for (int z = 0; z < inputs.Depth; z++)
                for (int y = 0; y < inputs.Height; y++)
                    for (int x = 0; x < inputs.Width; x++)
                    {
                        int px = x / sampleWidth;
                        int py = y / sampleHeight;

                        if (outputs[px, py, z] < inputs[x, y, z])
                        {
                            outputs[px, py, z] = inputs[x, y, z];
                            switches[px, py, z] = (x, y, z);
                        }
                    }

            return outputs;
        }

        public void Dump(BinaryWriter writer)
        {
        }

        public void Restore(BinaryReader reader)
        {
        }

        public class Builder : ILayerBuilder
        {
            private int sampleHeight;
            private int sampleWidth;

            public ILayer Build(ILayer inputLayer)
            {
                return new MaxPoolingLayer(sampleWidth, sampleHeight, inputLayer);
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