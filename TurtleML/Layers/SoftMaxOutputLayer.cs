using System;
using System.IO;

namespace TurtleML.Layers
{
    public class SoftMaxOutputLayer : ILayer
    {
        private readonly ILayer inputLayer;
        private readonly Tensor outputs;

        private SoftMaxOutputLayer(ILayer inputLayer)
        {
            this.inputLayer = inputLayer ?? throw new ArgumentNullException(nameof(inputLayer));

            var inputs = inputLayer.Outputs;
            var inputSize = inputs.Length;

            outputs = new Tensor(inputSize);
        }

        public ILayer InputLayer => inputLayer;

        public Tensor Outputs => outputs;

        public int OutputSize => outputs.Length;

        public Tensor Backpropagate(Tensor errors, float learningRate)
        {
            Tensor signals = new Tensor(outputs.Length);

            for (int o = 0; o < outputs.Length; o++)
            {
                float error = errors[o];
                float output = outputs[o];
                float derivative = (1f - output) * output;

                signals[o] = error * derivative;
            }

            return signals;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            float sum = 0f;
            for (int i = 0; i < inputs.Length; i++)
                sum += (float)Math.Exp(inputs[i]);

            for (int i = 0; i < inputs.Length; i++)
                outputs[i] = (float)Math.Exp(inputs[i]) / sum;

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
            public ILayer Build(ILayer inputLayer)
            {
                return new SoftMaxOutputLayer(inputLayer);
            }
        }
    }
}