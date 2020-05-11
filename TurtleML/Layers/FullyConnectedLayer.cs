using System;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace TurtleML.Layers
{
    public sealed class FullyConnectedLayer : ILayer
    {
        private readonly IActivationFunction activation;
        private readonly float[] bias;
        private readonly ThreadLocal<Tensor> buffers = new ThreadLocal<Tensor>();
        private readonly ILayer inputLayer;
        private readonly int inputSize;
        private readonly float[] momentum;
        private readonly int outputSize;
        private readonly Tensor signals;
        private readonly Tensor[] weights;
        private readonly Tensor derivatives;

        private FullyConnectedLayer(int outputSize, IActivationFunction activation, ILayer inputLayer)
        {
            if (outputSize < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(outputSize), "output size must be greater than zero");
            }

            this.outputSize = outputSize;
            this.activation = activation ?? throw new ArgumentNullException(nameof(activation));
            this.inputLayer = inputLayer ?? throw new ArgumentNullException(nameof(inputLayer));

            var inputs = inputLayer.Outputs;
            inputSize = inputs.Length;

            bias = new float[outputSize];
            momentum = new float[outputSize];
            Outputs = new Tensor(outputSize);
            derivatives = new Tensor(outputSize);
            signals = new Tensor(inputs.Dimensions);
            weights = new Tensor[outputSize];
            for (int w = 0; w < outputSize; w++)
            {
                weights[w] = new Tensor(inputSize);
            }
        }

        public Tensor Outputs { get; }

        public Tensor Backpropagate(Tensor errors, float learningRate, float momentumRate)
        {
            var inputs = inputLayer.Outputs;

            signals.Clear();

            for (int d = 0, count = Outputs.Length; d < count; d++)
            {
                derivatives[d] = activation.Derivative(Outputs[d]);
            }

            var gradients = derivatives.Multiply(errors);

            for (int o = 0; o < errors.Length; o++)
            {
                float gradient = gradients[o];

                signals.Add(Tensor.Multiply(weights[o], gradient));

                float delta = gradient * learningRate;
                float force = (momentumRate * momentum[o]) + delta;

                momentum[o] = delta;

                weights[o].Add(Tensor.Multiply(inputs, force));

                bias[o] += force;
            }

            return signals;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            Debug.Assert(inputs.Length == inputSize, $"Your input array (size: {inputs.Length}) does not match the specified size of {inputSize}.");

            for (int o = 0; o < outputSize; o++)
            {
                Outputs[o] = activation.Activate(Tensor.Dot(inputs, weights[o]) + bias[o]);
            }

            return Outputs;
        }

        public void Dump(BinaryWriter writer)
        {
            writer.Write(inputSize * outputSize);

            for (int o = 0; o < outputSize; o++)
            {
                for (int w = 0; w < inputSize; w++)
                {
                    writer.Write(weights[o][w]);
                }

                writer.Write(bias[o]);
            }
        }

        public void Initialize(Random random)
        {
            var rnd = random ?? new Random();
            var limit = 1f / inputSize;
            for (int o = 0; o < outputSize; o++)
            {
                for (int w = 0; w < inputSize; w++)
                {
                    weights[o][w] = ((float)rnd.NextDouble() * (limit + limit)) - limit;
                }

                bias[o] = ((float)rnd.NextDouble() * (limit + limit)) - limit;
            }
        }

        public void Restore(BinaryReader reader)
        {
            int count = reader.ReadInt32();

            Debug.Assert(count == inputSize * outputSize, $"Attempting to restore {nameof(FullyConnectedLayer)} with mismatched sizes.");

            for (int o = 0; o < outputSize; o++)
            {
                for (int w = 0; w < inputSize; w++)
                {
                    weights[o][w] = reader.ReadSingle();
                }

                bias[o] = reader.ReadSingle();
            }
        }

        public class Builder : ILayerBuilder
        {
            private IActivationFunction activation;
            private int outputCount;

            public Builder Activation(IActivationFunction activation)
            {
                this.activation = activation ?? throw new ArgumentNullException(nameof(activation));

                return this;
            }

            public ILayer Build(ILayer inputLayer)
            {
                return new FullyConnectedLayer(outputCount, activation, inputLayer);
            }

            public Builder Outputs(int outputCount)
            {
                if (outputCount < 1)
                {
                    throw new ArgumentOutOfRangeException("output size must be greater than zero", nameof(outputCount));
                }

                this.outputCount = outputCount;

                return this;
            }
        }
    }
}