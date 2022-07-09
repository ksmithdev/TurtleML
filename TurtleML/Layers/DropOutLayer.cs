using System;
using System.Diagnostics;
using System.IO;

namespace TurtleML.Layers
{
    public sealed class DropOutLayer : ILayer
    {
        private readonly float dropOut;
        private Random random;

        private DropOutLayer(float dropOut, IOutput input)
        {
            if (dropOut < float.Epsilon || dropOut > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(dropOut), "drop out must be between zero and one.");
            }

            this.dropOut = dropOut;

            var inputs = input.Outputs;
            Outputs = new Tensor(inputs.Dimensions);
        }

        public Tensor Outputs { get; }

        public Tensor Backpropagate(Tensor errors, float learningRate, float momentumRate)
        {
            return errors;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            Debug.Assert(inputs.Length == Outputs.Length, $"Your input array (size: {inputs.Length}) does not match the specified size of {Outputs.Length}.");

            if (!training)
            {
                return inputs;
            }

            for (int h = 0, count = Outputs.Length; h < count; h++)
            {
                Outputs[h] = random.NextDouble() >= dropOut ? inputs[h] : 0f;
            }

            return Outputs;
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

            public ILayer Build(IOutput input)
            {
                return new DropOutLayer(dropOut, input);
            }

            public Builder DropOut(float dropOut)
            {
                this.dropOut = dropOut;

                return this;
            }
        }
    }
}