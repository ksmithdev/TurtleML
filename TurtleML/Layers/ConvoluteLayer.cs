using System;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace TurtleML.Layers
{
    public class ConvoluteLayer : ILayer
    {
        private readonly IActivationFunction activation;
        private readonly float[] bias;
        private readonly ThreadLocal<Tensor> buffers = new ThreadLocal<Tensor>();
        private readonly int featureHeight;
        private readonly int featureSize;
        private readonly int featureWidth;
        private readonly int filterDepth;
        private readonly int filterHeight;
        private readonly int filterStride;
        private readonly int filterWidth;
        private readonly int inputDepth;
        private readonly ILayer inputLayer;
        private readonly float[] momentum;
        private readonly float momentumRate;
        private readonly Tensor outputs;
        private readonly Tensor signals;
        private readonly Tensor[] weights;

        private ConvoluteLayer(int filterWidth, int filterHeight, int filterStride, int filterDepth, float momentumRate, IActivationFunction activation, ILayer inputLayer)
        {
            this.activation = activation;
            this.filterWidth = filterWidth;
            this.filterHeight = filterHeight;
            this.filterStride = filterStride;
            this.filterDepth = filterDepth;
            this.momentumRate = momentumRate;
            this.inputLayer = inputLayer;

            var inputs = inputLayer.Outputs;
            //if (inputs.Dimensions.Length < 3)
            //    throw new ArgumentException("input layer needs to output a 3 dimentional tensor", nameof(inputLayer));

            int inputWidth = inputs.Width;
            int inputHeight = inputs.Height;
            inputDepth = inputs.Depth;

            featureWidth = (inputWidth - filterWidth) / filterStride + 1;
            featureHeight = (inputHeight - filterHeight) / filterStride + 1;
            featureSize = featureWidth * featureHeight;

            outputs = new Tensor(featureWidth, featureHeight, filterDepth);

            bias = new float[filterDepth];
            momentum = new float[filterDepth];
            signals = new Tensor(inputWidth, inputHeight, inputDepth);
            weights = new Tensor[filterDepth];
            for (int f = 0; f < filterDepth; f++)
                weights[f] = new Tensor(filterWidth * filterHeight * inputDepth);
        }

        public Tensor Outputs => outputs;

        public Tensor Backpropagate(Tensor errors, float learningRate)
        {
            var inputs = inputLayer.Outputs;

            signals.Clear();

            var buffer = buffers.Value ?? (buffers.Value = new Tensor(filterWidth * filterHeight * inputDepth));

            for (int f = 0; f < errors.Depth; f++)
                for (int y = 0; y < errors.Height; y++)
                    for (int x = 0; x < errors.Width; x++)
                    {
                        float error = errors[x, y, f];
                        float derivative = activation.Derivative(outputs[x, y, f]);
                        float gradient = error * derivative;

                        for (int fz = 0; fz < inputDepth; fz++)
                            for (int fy = 0; fy < filterHeight; fy++)
                                for (int fx = 0; fx < filterWidth; fx++)
                                {
                                    int filterIndex = IndexOf(fx, fy, fz);

                                    buffer[filterIndex] = inputs[x + fx, y + fy, fz];
                                    signals[x + fx, y + fy, fz] += gradient * weights[f][filterIndex];
                                }

                        float delta = gradient * learningRate;
                        float force = momentumRate * momentum[f] + delta;

                        momentum[f] = delta;

                        weights[f].Add(buffer.Multiply(force));

                        bias[f] += force;
                    }

            return signals;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            var buffer = buffers.Value ?? (buffers.Value = new Tensor(filterWidth * filterHeight * inputDepth));

            for (int x = 0; x < featureWidth; x++)
                for (int y = 0; y < featureHeight; y++)
                {
                    for (int fz = 0; fz < inputDepth; fz++)
                        for (int fy = 0; fy < filterHeight; fy++)
                            for (int fx = 0; fx < filterWidth; fx++)
                                buffer[IndexOf(fx, fy, fz)] = inputs[x + fx, y + fy, fz];

                    for (int f = 0; f < filterDepth; f++)
                        outputs[x, y, f] = activation.Activate(Tensor.Dot(buffer, weights[f]) + bias[f]);
                }

            return outputs;
        }

        public void Dump(BinaryWriter writer)
        {
            writer.Write(weights.Length);

            for (int f = 0; f < filterDepth; f++)
            {
                for (int w = 0; w < weights[f].Length; w++)
                    writer.Write(weights[f][w]);
                writer.Write(bias[f]);
            }
        }

        public void Initialize(Random random)
        {
            var rnd = random ?? new Random();
            var limit = 1f / (filterWidth * filterHeight);
            for (int f = 0, features = filterDepth; f < features; f++)
            {
                for (int w = 0; w < weights[f].Length; w++)
                    weights[f][w] = (float)rnd.NextDouble() * (limit + limit) - limit;
                bias[f] = (float)rnd.NextDouble() * (limit + limit) - limit;
            }
        }

        public void Restore(BinaryReader reader)
        {
            int count = reader.ReadInt32();

            Debug.Assert(count == filterDepth, $"Attempting to restore {nameof(ConvoluteLayer)} with mismatched sizes.");

            for (int f = 0; f < filterDepth; f++)
            {
                for (int w = 0; w < weights[f].Length; w++)
                    weights[f][w] = reader.ReadSingle();

                bias[f] = reader.ReadSingle();
            }
        }

        private int IndexOf(int x, int y, int z)
        {
            return x + (y * filterWidth) + (z * filterWidth * filterHeight);
        }

        public class Builder : ILayerBuilder
        {
            private IActivationFunction activation;
            private int filterCount;
            private int filterHeight;
            private int filterStride;
            private int filterWidth;
            private float momentumRate = 0.9f;

            public Builder Activation(IActivationFunction activation)
            {
                this.activation = activation ?? throw new ArgumentNullException(nameof(activation));

                return this;
            }

            public ILayer Build(ILayer inputLayer)
            {
                if (filterWidth < 1 || filterHeight < 1 || filterStride < 1 || filterCount < 1)
                    throw new InvalidOperationException("filter size, stride, and count must be greater than zero");
                if (activation == null)
                    throw new InvalidOperationException("activation cannot be null");

                return new ConvoluteLayer(filterWidth, filterHeight, filterStride, filterCount, momentumRate, activation, inputLayer);
            }

            public Builder Filters(int width, int height, int stride, int count)
            {
                filterWidth = width;
                filterHeight = height;
                filterStride = stride;
                filterCount = count;

                return this;
            }

            public Builder MomentumRate(float momentumRate)
            {
                this.momentumRate = momentumRate;

                return this;
            }
        }
    }
}