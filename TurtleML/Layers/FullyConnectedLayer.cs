namespace TurtleML.Layers;

using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;

/// <summary>
/// Represents a fully connected (dense) neural network layer.
/// </summary>
public sealed class FullyConnectedLayer : ILayer
{
    private readonly IInitializer biasInitializer;
    private readonly Tensor derivatives;
    private readonly float[] momentum;
    private readonly Tensor signals;
    private readonly IInitializer weightInitializer;
    private IActivationFunction activation;
    private float[] bias;
    private int inputSize;
    private int outputSize;
    private Tensor[] weights;

    private FullyConnectedLayer(int outputSize, IActivationFunction activation, IInitializer weightInitializer, IInitializer biasInitializer, IOutput input)
    {
        if (outputSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(outputSize), "output size must be greater than zero");
        }

        this.outputSize = outputSize;
        this.activation = activation ?? throw new ArgumentNullException(nameof(activation));
        this.weightInitializer = weightInitializer ?? throw new ArgumentNullException(nameof(weightInitializer));
        this.biasInitializer = biasInitializer ?? throw new ArgumentNullException(nameof(biasInitializer));

        var inputs = input.Outputs;
        inputSize = inputs.Length;

        Outputs = new Tensor(outputSize);

        bias = new float[outputSize];
        weights = new Tensor[outputSize];
        for (int w = 0; w < outputSize; w++)
        {
            weights[w] = new Tensor(inputSize);
        }

        momentum = new float[outputSize];
        derivatives = new Tensor(outputSize);
        signals = new Tensor(inputs.Dimensions);
    }

    /// <inheritdoc/>
    public Tensor Outputs { get; private set; }

    /// <inheritdoc/>
    public Tensor Backpropagate(Tensor inputs, Tensor errors, float learningRate, float momentumRate)
    {
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

    /// <inheritdoc/>
    public Tensor CalculateOutputs(Tensor inputs, bool training = false)
    {
        Debug.Assert(inputs.Length == inputSize, $"Your input array (size: {inputs.Length}) does not match the specified size of {inputSize}.");

        for (int o = 0; o < outputSize; o++)
        {
            Outputs[o] = activation.Activate(Tensor.Dot(inputs, weights[o]) + bias[o]);
        }

        return Outputs;
    }

    /// <inheritdoc/>
    public void Dump(BinaryWriter writer)
    {
        writer.Write(activation.GetType().AssemblyQualifiedName);
        activation.Dump(writer);

        writer.Write(inputSize);
        writer.Write(outputSize);

        writer.Write(Outputs.Width);
        writer.Write(Outputs.Height);
        writer.Write(Outputs.Depth);

        for (int o = 0; o < outputSize; o++)
        {
            for (int w = 0; w < inputSize; w++)
            {
                writer.Write(weights[o][w]);
            }

            writer.Write(bias[o]);
        }
    }

    /// <inheritdoc/>
    public void Initialize(Random random)
    {
        var rnd = random ?? new Random();

        for (int o = 0; o < outputSize; o++)
        {
            for (int w = 0; w < inputSize; w++)
            {
                weights[o][w] = weightInitializer.Sample(inputSize, outputSize, rnd);
            }

            bias[o] = biasInitializer.Sample(inputSize, outputSize, rnd);
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

        inputSize = reader.ReadInt32();
        outputSize = reader.ReadInt32();

        int outputWidth = reader.ReadInt32();
        int outputHeight = reader.ReadInt32();
        int outputDepth = reader.ReadInt32();

        Outputs = new Tensor(outputWidth, outputHeight, outputDepth);

        bias = new float[outputSize];
        weights = new Tensor[outputSize];
        for (int w = 0; w < outputSize; w++)
        {
            weights[w] = new Tensor(inputSize);
        }

        for (int o = 0; o < outputSize; o++)
        {
            for (int w = 0; w < inputSize; w++)
            {
                weights[o][w] = reader.ReadSingle();
            }

            bias[o] = reader.ReadSingle();
        }
    }

    /// <summary>
    /// Builder class for constructing <see cref="FullyConnectedLayer"/> instances.
    /// </summary>
    public class Builder : ILayerBuilder
    {
        private IActivationFunction? activation;
        private IInitializer? biasInitializer;
        private int outputCount;
        private IInitializer? weightInitializer;

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
            if (activation == null)
            {
                throw new InvalidOperationException($"{nameof(activation)} cannot be null.");
            }

            if (weightInitializer == null)
            {
                throw new InvalidOperationException($"{nameof(weightInitializer)} cannot be null.");
            }

            if (biasInitializer == null)
            {
                throw new InvalidOperationException($"{nameof(biasInitializer)} cannot be null.");
            }

            return new FullyConnectedLayer(outputCount, activation, weightInitializer, biasInitializer, input);
        }

        /// <summary>
        /// Sets weight and bias initializers for the layer.
        /// </summary>
        /// <param name="weight">Initializer for weights.</param>
        /// <param name="bias">Optional initializer for biases.</param>
        /// <returns>The current builder instance.</returns>
        public Builder Initializer(IInitializer weight, IInitializer? bias = null)
        {
            weightInitializer = weight ?? throw new ArgumentNullException(nameof(weight));
            biasInitializer = bias ?? weight ?? throw new ArgumentNullException(nameof(bias));

            return this;
        }

        /// <summary>
        /// Sets the number of output neurons.
        /// </summary>
        /// <param name="outputCount">The number of output neurons in the layer.</param>
        /// <returns>The current builder instance.</returns>
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