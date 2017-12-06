using System;
using System.IO;

namespace TurtleML.Layers
{
    public class InputLayer : ILayer
    {
        private readonly Tensor outputs;

        private InputLayer(int width, int height, int depth, ILayer inputLayer)
        {
            outputs = new Tensor(width, height, depth);
        }

        public Tensor Outputs => outputs;

        public Tensor Backpropagate(Tensor errors, float learningRate)
        {
            return errors;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            outputs.Load(inputs);

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
            private int depth = 1;
            private int height = 1;
            private int width = 1;

            public ILayer Build(ILayer inputLayer)
            {
                return new InputLayer(width, height, depth, inputLayer);
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