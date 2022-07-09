using System;
using System.IO;

namespace TurtleML.Layers
{
    public sealed class InputLayer : ILayer
    {
        private InputLayer()
        {
        }

        public Tensor Outputs { get; private set; }

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
            writer.Write(Outputs.Width);
            writer.Write(Outputs.Length);
            writer.Write(Outputs.Depth);
        }

        public void Initialize(Random random)
        {
        }

        public void Restore(BinaryReader reader)
        {
            int width = reader.ReadInt32();
            int length = reader.ReadInt32();
            int depth = reader.ReadInt32();

            Outputs = new Tensor((width, length, depth));
        }

        public class Builder : ILayerBuilder
        {
            private int depth = 1;
            private int height = 1;
            private int width = 1;

            public ILayer Build(IOutput input)
            {
                return new InputLayer() { Outputs = new Tensor((width, height, depth)) };
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

            public Builder Dimensions(int length)
            {
                this.width = length;

                return this;
            }
        }
    }
}