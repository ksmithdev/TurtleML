using System;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace TurtleML.Layers
{
    public sealed class ConvoluteLayer : ILayer
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
        private readonly IInitializer initializer;
        private readonly IOutput input;
        private readonly int inputDepth;
        private readonly float[] momentum;
        private readonly Tensor signals;
        private readonly Tensor[] weights;

        private ConvoluteLayer(int filterWidth, int filterHeight, int filterStride, int filterDepth, IActivationFunction activation, IInitializer initializer, IOutput inputLayer)
        {
            this.activation = activation ?? throw new ArgumentNullException(nameof(activation));
            this.initializer = initializer ?? throw new ArgumentNullException(nameof(initializer));
            this.filterWidth = filterWidth;
            this.filterHeight = filterHeight;
            this.filterStride = filterStride;
            this.filterDepth = filterDepth;
            this.input = inputLayer;

            var inputs = inputLayer.Outputs;
            int inputWidth = inputs.Width;
            int inputHeight = inputs.Height;
            inputDepth = inputs.Depth;

            featureWidth = ((inputWidth - filterWidth) / filterStride) + 1;
            featureHeight = ((inputHeight - filterHeight) / filterStride) + 1;
            featureSize = featureWidth * featureHeight;

            Outputs = new Tensor(featureWidth, featureHeight, filterDepth);

            bias = new float[filterDepth];
            momentum = new float[filterDepth];
            signals = new Tensor(inputWidth, inputHeight, inputDepth);
            weights = new Tensor[filterDepth];
            for (int f = 0; f < filterDepth; f++)
            {
                weights[f] = new Tensor(filterWidth * filterHeight * inputDepth);
            }
        }

        public Tensor Outputs { get; private set; }

        public Tensor Backpropagate(Tensor errors, float learningRate, float momentumRate)
        {
            var inputs = input.Outputs;

            signals.Clear();

            var buffer = buffers.Value ?? (buffers.Value = new Tensor(filterWidth, filterHeight, inputDepth));

            var derivatives = new Tensor(Outputs.Dimensions);
            for (int d = 0, count = Outputs.Length; d < count; d++)
            {
                derivatives[d] = activation.Derivative(Outputs[d]);
            }

            var gradients = derivatives.Multiply(errors);

            // TODO: some way to speed up signal calculations? this is the slowest part of the whole system right now
            //for (int f = 0; f < signals.Depth; f++)
            //    for (int y = 0; y < signals.Height; y++)
            //        for (int x = 0; x < signals.Width; x++)
            //        {
            //        }

            for (int f = 0, depth = gradients.Depth; f < depth; f++)
            {
                for (int y = 0, height = gradients.Height; y < height; y += filterStride)
                {
                    for (int x = 0, width = gradients.Width; x < width; x += filterStride)
                    {
                        float gradient = gradients[x, y, f];

                        Tensor.Multiply(weights[f], gradient, buffer);

                        for (int fz = 0; fz < inputDepth; fz++)
                        {
                            for (int fy = 0; fy < filterHeight; fy++)
                            {
                                for (int fx = 0; fx < filterWidth; fx++)
                                {
                                    int bufferIndex = buffer.IndexOf(fx, fy, fz);
                                    int inputIndex = inputs.IndexOf(x + fx, y + fy, fz);

                                    signals[inputIndex] += buffer[bufferIndex];

                                    buffer[bufferIndex] = inputs[inputIndex];
                                }
                            }
                        }

                        float delta = gradient * learningRate;
                        float force = (momentumRate * momentum[f]) + delta;

                        momentum[f] = delta;

                        weights[f].Add(buffer.Multiply(force));

                        bias[f] += force;
                    }
                }
            }

            return signals;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            var buffer = buffers.Value ?? (buffers.Value = new Tensor(filterWidth, filterHeight, inputDepth));

            for (int x = 0; x < featureWidth; x++)
            {
                for (int y = 0; y < featureHeight; y++)
                {
                    for (int fz = 0; fz < inputDepth; fz++)
                    {
                        for (int fy = 0; fy < filterHeight; fy++)
                        {
                            for (int fx = 0; fx < filterWidth; fx++)
                            {
                                int inputIndex = inputs.IndexOf(x + fx, y + fy, fz);
                                int bufferIndex = buffer.IndexOf(fx, fy, fz);

                                buffer[bufferIndex] = inputs[inputIndex];
                            }
                        }
                    }

                    for (int f = 0; f < filterDepth; f++)
                    {
                        Outputs[x, y, f] = activation.Activate(Tensor.Dot(buffer, weights[f]) + bias[f]);
                    }
                }
            }

            return Outputs;
        }

        public void Dump(BinaryWriter writer)
        {
            writer.Write(weights.Length);

            for (int f = 0; f < filterDepth; f++)
            {
                for (int w = 0; w < weights[f].Length; w++)
                {
                    writer.Write(weights[f][w]);
                }

                writer.Write(bias[f]);
            }
        }

        public void Initialize(Random random)
        {
            var rnd = random ?? new Random();
            var inputs = filterWidth * filterHeight;
            var outputs = Outputs.Length;
            for (int f = 0, features = filterDepth; f < features; f++)
            {
                for (int w = 0; w < weights[f].Length; w++)
                    weights[f][w] = initializer.Sample(inputs, outputs, rnd);

                bias[f] = initializer.Sample(inputs, outputs, rnd);
            }
        }

        public void Restore(BinaryReader reader)
        {
            int count = reader.ReadInt32();

            Debug.Assert(count == filterDepth, $"Attempting to restore {nameof(ConvoluteLayer)} with mismatched sizes.");

            for (int f = 0; f < filterDepth; f++)
            {
                for (int w = 0; w < weights[f].Length; w++)
                {
                    weights[f][w] = reader.ReadSingle();
                }

                bias[f] = reader.ReadSingle();
            }
        }

        public class Builder : ILayerBuilder
        {
            private IActivationFunction activation;
            private int filterCount;
            private int filterHeight;
            private int filterStride;
            private int filterWidth;
            private IInitializer initializer;

            public Builder Activation(IActivationFunction activation)
            {
                this.activation = activation ?? throw new ArgumentNullException(nameof(activation));

                return this;
            }

            public ILayer Build(IOutput input)
            {
                if (filterWidth < 1 || filterHeight < 1 || filterStride < 1 || filterCount < 1)
                {
                    throw new InvalidOperationException("filter size, stride, and count must be greater than zero");
                }

                if (activation == null)
                {
                    throw new InvalidOperationException("activation cannot be null");
                }

                return new ConvoluteLayer(filterWidth, filterHeight, filterStride, filterCount, activation, initializer, input);
            }

            public Builder Filters(int width, int height, int stride, int count)
            {
                filterWidth = width;
                filterHeight = height;
                filterStride = stride;
                filterCount = count;

                return this;
            }

            public Builder Initializer(IInitializer initializer)
            {
                this.initializer = initializer ?? throw new ArgumentNullException(nameof(initializer));

                return this;
            }
        }
    }
}