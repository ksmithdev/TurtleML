using System;
using System.IO;

namespace TurtleML.Layers
{
    public sealed class SoftMaxOutputLayer : ILayer
    {
        private SoftMaxOutputLayer(ILayer inputLayer)
        {
            InputLayer = inputLayer ?? throw new ArgumentNullException(nameof(inputLayer));

            var inputs = inputLayer.Outputs;
            var inputSize = inputs.Length;

            Outputs = new Tensor(inputSize);
        }

        public ILayer InputLayer { get; }

        public Tensor Outputs { get; }

        public int OutputSize => Outputs.Length;

        public Tensor Backpropagate(Tensor errors, float learningRate, float momentumRate)
        {
            var signals = new Tensor(Outputs.Length);

            for (int o = 0; o < Outputs.Length; o++)
            {
                float error = errors[o];
                float output = Outputs[o];
                float derivative = (1f - output) * output;

                signals[o] = error * derivative;
            }

            return signals;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            float sum = 0f;
            for (int i = 0; i < inputs.Length; i++)
            {
                sum += (float)Math.Exp(inputs[i]);
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                Outputs[i] = (float)Math.Exp(inputs[i]) / sum;
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
            public ILayer Build(ILayer inputLayer)
            {
                return new SoftMaxOutputLayer(inputLayer);
            }
        }
    }
}