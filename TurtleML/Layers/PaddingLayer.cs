namespace TurtleML.Layers;

using System;
using System.IO;

/// <summary>
/// Adds padding to the input tensor.
/// </summary>
public class PaddingLayer : ILayer
{
    private readonly Tensor signals = Tensor.Empty;
    private int padding;

    /// <summary>
    /// Initializes a new instance of the <see cref="PaddingLayer"/> class.
    /// </summary>
    /// <param name="padding">The amount of padding to apply.</param>
    /// <param name="input">The input tensor to this layer.</param>
    public PaddingLayer(int padding, IOutput input)
    {
        this.padding = padding;

        var inputs = input.Outputs;
        int width = inputs.Width + (padding * 2);
        int height = inputs.Height + (padding * 2);
        int depth = inputs.Depth;

        Outputs = new Tensor(width, height, depth);
        signals = new Tensor(inputs.Dimensions);
    }

    private PaddingLayer()
    {
    }

    /// <inheritdoc/>
    public Tensor Outputs { get; private set; } = Tensor.Empty;

    /// <inheritdoc/>
    public Tensor Backpropagate(Tensor inputs, Tensor errors, float learningRate, float momentumRate)
    {
        int width = signals.Width;
        for (int z = 0, depth = signals.Depth; z < depth; z++)
        {
            for (int y = 0, height = signals.Height; y < height; y++)
            {
                Tensor.Copy(errors, padding, y + padding, z, signals, 0, y, z, width);

                // old way
                // for (int x = 0, width = signals.Width; x < width; x++)
                //    signals[x, y, z] = errors[x + padding, y + padding, z];
            }
        }

        return signals;
    }

    /// <inheritdoc/>
    public Tensor CalculateOutputs(Tensor inputs, bool training = false)
    {
        int width = inputs.Width;
        for (int z = 0, depth = inputs.Depth; z < depth; z++)
        {
            for (int y = 0, height = inputs.Height; y < height; y++)
            {
                Tensor.Copy(inputs, 0, y, z, Outputs, padding, y + padding, z, width);

                // old way
                // for (int x = 0; x < width; x++)
                //    outputs[x + padding, y + padding, z] = inputs[x, y, z];
            }
        }

        return Outputs;
    }

    /// <inheritdoc/>
    public void Dump(BinaryWriter writer)
    {
        writer.Write(padding);

        writer.Write(Outputs.Width);
        writer.Write(Outputs.Height);
        writer.Write(Outputs.Depth);
    }

    /// <inheritdoc/>
    public void Initialize(Random random)
    {
    }

    /// <inheritdoc/>
    public void Restore(BinaryReader reader)
    {
        padding = reader.ReadInt32();

        int width = reader.ReadInt32();
        int height = reader.ReadInt32();
        int depth = reader.ReadInt32();

        Outputs = new Tensor(width, height, depth);
    }

    /// <summary>
    /// Builder class for constructing <see cref="PaddingLayer"/> instances.
    /// </summary>
    public class Builder : ILayerBuilder
    {
        private int size;

        /// <inheritdoc/>
        public ILayer Build(IOutput input) => new PaddingLayer(size, input);

        /// <summary>
        /// Sets the padding size.
        /// </summary>
        /// <param name="size">The amount of padding to apply.</param>
        /// <returns>The current builder instance.</returns>
        public Builder Padding(int size)
        {
            this.size = size;

            return this;
        }
    }
}