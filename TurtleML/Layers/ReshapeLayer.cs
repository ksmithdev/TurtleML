using System;
using System.IO;

namespace TurtleML.Layers
{
    public sealed class ReshapeLayer : ILayer
    {
        private ReshapeLayer()
        {
        }

        public Tensor Outputs { get; private set; } = Tensor.Empty;

        public Tensor Backpropagate(Tensor? inputs, Tensor errors, float learningRate, float momentumRate)
        {
            return errors;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            if (inputs.Length != Outputs.Length)
            {
                throw new InvalidOperationException($"Input of shape ({inputs.Width},{inputs.Length},{inputs.Depth}) cannot be reshaped into ({Outputs.Width},{Outputs.Length},{Outputs.Depth})");
            }

            Outputs.Load(inputs.Reshape(Outputs.Width, Outputs.Length, Outputs.Depth));

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

            Outputs = new Tensor(width, length, depth);
        }

        public class Builder : ILayerBuilder
        {
            private int depth = 1;
            private int length = 1;
            private int width = 1;

            public ILayer Build(IOutput input)
            {
                return new ReshapeLayer() { Outputs = new Tensor((width, length, depth)) };
            }

            public Builder Dimensions(int width, int length, int depth)
            {
                this.width = width;
                this.length = length;
                this.depth = depth;

                return this;
            }

            public Builder Dimensions(int width, int height)
            {
                this.width = width;
                this.length = height;

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