namespace TurtleML.Layers;

using System;
using System.IO;
using System.Runtime.CompilerServices;
using TurtleML.Initializers;

/// <summary>
/// Represents a convolutional layer in a neural network, applying filters to input data.
/// </summary>
public sealed class ConvoluteLayer : ILayer
{
    private readonly float[] momentum;
    private readonly Tensor signals;
    private IActivationFunction activation;
    private float[] bias;
    private int featureHeight;
    private int featureSize;
    private int featureWidth;
    private int filterDepth;
    private int filterHeight;
    private int filterStride;
    private int filterWidth;
    private IInitializer initializer;
    private Tensor[] weights;

    private ConvoluteLayer(int filterWidth, int filterHeight, int filterStride, int filterDepth, IActivationFunction activation, IInitializer initializer, IOutput input)
    {
        this.activation = activation ?? throw new ArgumentNullException(nameof(activation));
        this.initializer = initializer ?? throw new ArgumentNullException(nameof(initializer));
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.filterStride = filterStride;
        this.filterDepth = filterDepth;

        var inputs = input.Outputs;
        int inputWidth = inputs.Width;
        int inputHeight = inputs.Height;
        int inputDepth = inputs.Depth;

        featureWidth = ((inputWidth - filterWidth) / filterStride) + 1;
        featureHeight = ((inputHeight - filterHeight) / filterStride) + 1;
        featureSize = featureWidth * featureHeight;

        Outputs = new Tensor(featureWidth, featureHeight, filterDepth);

        bias = new float[filterDepth];
        weights = new Tensor[filterDepth];
        for (int f = 0; f < filterDepth; f++)
        {
            weights[f] = new Tensor(filterWidth * filterHeight * inputDepth);
        }

        momentum = new float[filterDepth];
        signals = new Tensor(inputWidth, inputHeight, inputDepth);
    }

    /// <inheritdoc/>
    public Tensor Outputs { get; private set; }

    /// <inheritdoc/>
    public Tensor Backpropagate(Tensor inputs, Tensor errors, float learningRate, float momentumRate)
    {
        signals.Clear();

        var buffer = new Tensor(filterWidth, filterHeight, inputs.Depth);

        var derivatives = new Tensor(Outputs.Dimensions);
        for (int d = 0, count = Outputs.Length; d < count; d++)
        {
            derivatives[d] = activation.Derivative(Outputs[d]);
        }

        var gradients = derivatives.Multiply(errors);

        /*
         * TODO: some way to speed up signal calculations ? this is the slowest part of the whole system right now
         * for (int f = 0; f < signals.Depth; f++)
         *     for (int y = 0; y < signals.Height; y++)
         *         for (int x = 0; x < signals.Width; x++)
         *         {
         *         }
        */

        for (int f = 0, depth = gradients.Depth; f < depth; f++)
        {
            for (int y = 0, height = gradients.Height; y < height; y += filterStride)
            {
                for (int x = 0, width = gradients.Width; x < width; x += filterStride)
                {
                    float gradient = gradients[x, y, f];

                    Tensor.Multiply(weights[f], gradient, buffer);

                    for (int fz = 0; fz < inputs.Depth; fz++)
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

    /// <inheritdoc/>
    public Tensor CalculateOutputs(Tensor inputs, bool training = false)
    {
        var buffer = new Tensor(filterWidth, filterHeight, inputs.Depth);

        for (int x = 0; x < featureWidth; x++)
        {
            for (int y = 0; y < featureHeight; y++)
            {
                for (int fz = 0; fz < inputs.Depth; fz++)
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

    /// <inheritdoc/>
    public void Dump(BinaryWriter writer)
    {
        writer.Write(activation.GetType().AssemblyQualifiedName);
        activation.Dump(writer);

        writer.Write(filterWidth);
        writer.Write(filterHeight);
        writer.Write(filterDepth);
        writer.Write(filterStride);

        writer.Write(featureWidth);
        writer.Write(featureHeight);
        writer.Write(featureSize);

        writer.Write(Outputs.Width);
        writer.Write(Outputs.Height);
        writer.Write(Outputs.Depth);

        for (int f = 0; f < filterDepth; f++)
        {
            writer.Write(weights[f].Length);

            for (int w = 0; w < weights[f].Length; w++)
            {
                writer.Write(weights[f][w]);
            }

            writer.Write(bias[f]);
        }
    }

    /// <inheritdoc/>
    public void Initialize(Random random)
    {
        var rnd = random ?? new Random();
        var inputs = filterWidth * filterHeight;
        var outputs = Outputs.Length;
        for (int f = 0, features = filterDepth; f < features; f++)
        {
            for (int w = 0; w < weights[f].Length; w++)
            {
                weights[f][w] = initializer.Sample(inputs, outputs, rnd);
            }

            bias[f] = initializer.Sample(inputs, outputs, rnd);
        }
    }

    /// <inheritdoc/>
    public void Restore(BinaryReader reader)
    {
        var activationType = reader.ReadString();
        if (RuntimeHelpers.GetUninitializedObject(Type.GetType(activationType)) is not IActivationFunction activation)
        {
            throw new InvalidOperationException($"An invalid activation \"{activationType}\" was specified.");
        }

        this.activation = activation;
        activation.Restore(reader);

        filterWidth = reader.ReadInt32();
        filterHeight = reader.ReadInt32();
        filterDepth = reader.ReadInt32();
        filterStride = reader.ReadInt32();

        featureWidth = reader.ReadInt32();
        featureHeight = reader.ReadInt32();
        featureSize = reader.ReadInt32();

        int outputWidth = reader.ReadInt32();
        int outputHeight = reader.ReadInt32();
        int outputDepth = reader.ReadInt32();

        Outputs = new Tensor(outputWidth, outputHeight, outputDepth);

        bias = new float[filterDepth];
        weights = new Tensor[filterDepth];

        for (int f = 0; f < filterDepth; f++)
        {
            int inputSize = reader.ReadInt32();

            weights[f] = new Tensor(inputSize);

            for (int w = 0; w < weights[f].Length; w++)
            {
                weights[f][w] = reader.ReadSingle();
            }

            bias[f] = reader.ReadSingle();
        }
    }

    /// <summary>
    /// Builder class for constructing <see cref="DropOutLayer"/> instances.
    /// </summary>
    public class Builder : ILayerBuilder
    {
        private IActivationFunction? activation;
        private int filterCount;
        private int filterHeight;
        private int filterStride;
        private int filterWidth;
        private IInitializer? initializer;

        /// <summary>
        /// Initializes a new instance of the <see cref="Builder"/> class.
        /// </summary>
        public Builder()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Builder"/> class.
        /// </summary>
        /// <param name="activation">The activation function for the layer.</param>
        /// <param name="filterWidth">Width of convolutional filters.</param>
        /// <param name="filterHeight">Height of convolutional filters.</param>
        /// <param name="filterStride">Strides between filter applications.</param>
        /// <param name="filterCount">Number of filters in the layer.</param>
        /// <param name="initializer">Optional weight initializer for the layer.</param>
        public Builder(IActivationFunction activation, int filterWidth, int filterHeight, int filterStride, int filterCount, IInitializer? initializer = null)
        {
            this.activation = activation ?? throw new ArgumentNullException(nameof(activation));
            this.filterWidth = filterWidth;
            this.filterHeight = filterHeight;
            this.filterStride = filterStride;
            this.filterCount = filterCount;
            this.initializer = initializer;
        }

        /// <summary>
        /// Sets the activation function for the layer.
        /// </summary>
        /// <param name="activation">The activation function to use.</param>
        /// <returns>The current builder instance.</returns>
        public Builder Activation(IActivationFunction activation)
        {
            this.activation = activation ?? throw new ArgumentNullException(nameof(activation));

            return this;
        }

        /// <inheritdoc/>
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

            return new ConvoluteLayer(filterWidth, filterHeight, filterStride, filterCount, activation, initializer ?? new HeInitializer(), input);
        }

        /// <summary>
        /// Configures filter dimensions, stride, and count for the layer.
        /// </summary>
        /// <param name="width">Width of convolutional filters.</param>
        /// <param name="height">Height of convolutional filters.</param>
        /// <param name="stride">Strides between filter applications.</param>
        /// <param name="count">Number of filters in the layer.</param>
        /// <returns>The current builder instance.</returns>
        public Builder Filters(int width, int height, int stride, int count)
        {
            filterWidth = width;
            filterHeight = height;
            filterStride = stride;
            filterCount = count;

            return this;
        }

        /// <summary>
        /// Sets the weight initializer for the layer.
        /// </summary>
        /// <param name="initializer">The weight initializer to use.</param>
        /// <returns>The current builder instance.</returns>
        public Builder Initializer(IInitializer initializer)
        {
            this.initializer = initializer ?? throw new ArgumentNullException(nameof(initializer));

            return this;
        }
    }
}