using System;
using System.IO;

namespace TurtleML.Layers
{
    public sealed class InputLayer : ILayer
    {
        private InputLayer(int width, int height, int depth)
        {
            Outputs = new Tensor(width, height, depth);
        }

        public Tensor Outputs { get; }

        public Tensor Backpropagate(Tensor errors, float learningRate, float momentumRate)
        {
            return errors;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            Outputs.Load(inputs);

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
            private int depth = 1;
            private int height = 1;
            private int width = 1;

            public ILayer Build(ILayer inputLayer)
            {
                return new InputLayer(width, height, depth);
            }

            public Builder Dimensions(int width, int height, int depth)
            {
                this.width = width;
                this.height = height;
                this.depth = depth;

                return this;
            }

            public Builder Dimensions(int width, int height)
            {
                this.width = width;
                this.height = height;

                return this;
            }
        }
    }
}