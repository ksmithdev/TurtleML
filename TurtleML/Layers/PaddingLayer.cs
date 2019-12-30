using System;
using System.IO;

namespace TurtleML.Layers
{
    public class PaddingLayer : ILayer
    {
        private readonly ILayer inputLayer;
        private readonly Tensor outputs;
        private readonly int padding;
        private readonly Tensor signals;

        public PaddingLayer(int padding, ILayer inputLayer)
        {
            this.padding = padding;
            this.inputLayer = inputLayer;

            var inputs = inputLayer.Outputs;
            int width = inputs.Width + padding * 2;
            int height = inputs.Height + padding * 2;
            int depth = inputs.Depth;

            outputs = new Tensor(width, height, depth);
            signals = new Tensor(inputs.Dimensions);
        }

        public Tensor Outputs => outputs;

        public Tensor Backpropagate(Tensor errors, float learningRate, float momentumRate)
        {
            int width = signals.Width;
            for (int z = 0, depth = signals.Depth; z < depth; z++)
                for (int y = 0, height = signals.Height; y < height; y++)
                {
                    Tensor.Copy(errors, padding, y + padding, z, signals, 0, y, z, width);

                    // old way
                    //for (int x = 0, width = signals.Width; x < width; x++)
                    //    signals[x, y, z] = errors[x + padding, y + padding, z];
                }

            return signals;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            int width = inputs.Width;
            for (int z = 0, depth = inputs.Depth; z < depth; z++)
                for (int y = 0, height = inputs.Height; y < height; y++)
                {
                    Tensor.Copy(inputs, 0, y, z, outputs, padding, y + padding, z, width);

                    // old way
                    //for (int x = 0; x < width; x++)
                    //    outputs[x + padding, y + padding, z] = inputs[x, y, z];
                }

            return outputs;
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
            private int size;

            public ILayer Build(ILayer inputLayer) => new PaddingLayer(size, inputLayer);

            public Builder Padding(int size)
            {
                this.size = size;

                return this;
            }
        }
    }
}