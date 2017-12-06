using System;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace TurtleML.Layers
{
    public class FullyConnectedLayer : ILayer
    {
        private readonly IActivationFunction activation;
        private readonly float[] bias;
        private readonly ThreadLocal<Tensor> buffers = new ThreadLocal<Tensor>();
        private readonly ILayer inputLayer;
        private readonly int inputSize;
        private readonly float[] momentum;
        private readonly float momentumRate;
        private readonly Tensor outputs;
        private readonly int outputSize;
        private readonly Tensor signals;
        private readonly Tensor[] weights;

        private FullyConnectedLayer(int outputSize, float momentumRate, IActivationFunction activation, ILayer inputLayer)
        {
            if (outputSize < 1)
                throw new ArgumentOutOfRangeException(nameof(outputSize), "output size must be greater than zero");
            if (momentumRate < float.Epsilon)
                throw new ArgumentOutOfRangeException(nameof(momentumRate), "momentum rate must be greater than zero");

            this.outputSize = outputSize;
            this.momentumRate = momentumRate;
            this.activation = activation ?? throw new ArgumentNullException(nameof(activation));
            this.inputLayer = inputLayer ?? throw new ArgumentNullException(nameof(inputLayer));

            var inputs = inputLayer.Outputs;
            inputSize = inputs.Length;

            bias = new float[outputSize];
            momentum = new float[outputSize];
            outputs = new Tensor(outputSize);
            signals = new Tensor(inputs.Dimensions);
            weights = new Tensor[outputSize];
            for (int w = 0; w < outputSize; w++)
                weights[w] = new Tensor(inputSize);
        }

        public Tensor Outputs => outputs;

        public Tensor Backpropagate(Tensor errors, float learningRate)
        {
            var inputs = inputLayer.Outputs;

            signals.Clear();

            for (int o = 0; o < errors.Length; o++)
            {
                float error = errors[o];
                float derivative = activation.Derivative(outputs[o]);
                float gradient = error * derivative;

                signals.Add(Tensor.Multiply(weights[o], gradient));

                float delta = gradient * learningRate;
                float force = momentumRate * momentum[o] + delta;

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
                outputs[o] = activation.Activate(Tensor.Dot(inputs, weights[o]) + bias[o]);

            return outputs;
        }

        public void Dump(BinaryWriter writer)
        {
            writer.Write(inputSize * outputSize);

            for (int o = 0; o < outputSize; o++)
            {
                for (int w = 0; w < inputSize; w++)
                    writer.Write(weights[o][w]);
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
                    weights[o][w] = (float)rnd.NextDouble() * (limit + limit) - limit;
                bias[o] = (float)rnd.NextDouble() * (limit + limit) - limit;
            }
        }

        public void Restore(BinaryReader reader)
        {
            int count = reader.ReadInt32();

            Debug.Assert(count == inputSize * outputSize, $"Attempting to restore {nameof(FullyConnectedLayer)} with mismatched sizes.");

            for (int o = 0; o < outputSize; o++)
            {
                for (int w = 0; w < inputSize; w++)
                    weights[o][w] = reader.ReadSingle();
                bias[o] = reader.ReadSingle();
            }
        }

        public class Builder : ILayerBuilder
        {
            private IActivationFunction activation;
            private float momentumRate = 0.9f;
            private int outputCount;

            public Builder Activation(IActivationFunction activation)
            {
                this.activation = activation ?? throw new ArgumentNullException(nameof(activation));

                return this;
            }

            public ILayer Build(ILayer inputLayer)
            {
                return new FullyConnectedLayer(outputCount, momentumRate, activation, inputLayer);
            }

            public Builder MomentumRate(float momentumRate)
            {
                this.momentumRate = momentumRate;

                return this;
            }

            public Builder Outputs(int outputCount)
            {
                if (outputCount < 1)
                    throw new ArgumentOutOfRangeException("output size must be greater than zero", nameof(outputCount));

                this.outputCount = outputCount;

                return this;
            }
        }
    }
}