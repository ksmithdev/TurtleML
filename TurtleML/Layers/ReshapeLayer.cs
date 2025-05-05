namespace TurtleML.Layers;

using System;
using System.IO;

/// <summary>
/// Reshapes the input tensor to a specified dimension.
/// </summary>
public sealed class ReshapeLayer : ILayer
{
    private ReshapeLayer()
    {
    }

    /// <inheritdoc/>
    public Tensor Outputs { get; private set; } = Tensor.Empty;

    /// <inheritdoc/>
    public Tensor Backpropagate(Tensor? inputs, Tensor errors, float learningRate, float momentumRate)
    {
        return errors;
    }

    /// <inheritdoc/>
    public Tensor CalculateOutputs(Tensor inputs, bool training = false)
    {
        if (inputs.Length != Outputs.Length)
        {
            throw new InvalidOperationException($"Input of shape ({inputs.Width},{inputs.Length},{inputs.Depth}) cannot be reshaped into ({Outputs.Width},{Outputs.Length},{Outputs.Depth})");
        }

        Outputs.Load(inputs.Reshape(Outputs.Width, Outputs.Length, Outputs.Depth));

        return Outputs;
    }

    /// <inheritdoc/>
    public void Dump(BinaryWriter writer)
    {
        writer.Write(Outputs.Width);
        writer.Write(Outputs.Length);
        writer.Write(Outputs.Depth);
    }

    /// <inheritdoc/>
    public void Initialize(Random random)
    {
    }

    /// <inheritdoc/>
    public void Restore(BinaryReader reader)
    {
        int width = reader.ReadInt32();
        int length = reader.ReadInt32();
        int depth = reader.ReadInt32();

        Outputs = new Tensor(width, length, depth);
    }

    /// <summary>
    /// Builder class for constructing <see cref="ReshapeLayer"/> instances.
    /// </summary>
    public class Builder : ILayerBuilder
    {
        private int depth = 1;
        private int length = 1;
        private int width = 1;

        /// <inheritdoc/>
        public ILayer Build(IOutput input)
        {
            return new ReshapeLayer() { Outputs = new Tensor((width, length, depth)) };
        }

        /// <summary>
        /// Sets the dimensions for reshaping.
        /// </summary>
        /// <param name="width">The new width of the tensor.</param>
        /// <param name="length">The new length of the tensor.</param>
        /// <param name="depth">The new depth of the tensor.</param>
        /// <returns>The current builder instance.</returns>
        public Builder Dimensions(int width, int length, int depth)
        {
            this.width = width;
            this.length = length;
            this.depth = depth;

            return this;
        }

        /// <summary>
        /// Sets the dimensions for reshaping.
        /// </summary>
        /// <param name="width">The new width of the tensor.</param>
        /// <param name="height">The new height of the tensor.</param>
        /// <returns>The current builder instance.</returns>
        public Builder Dimensions(int width, int height)
        {
            this.width = width;
            length = height;

            return this;
        }

        /// <summary>
        /// Sets the dimensions for reshaping.
        /// </summary>
        /// <param name="length">The new length of the tensor.</param>
        /// <returns>The current builder instance.</returns>
        public Builder Dimensions(int length)
        {
            width = length;

            return this;
        }
    }
}