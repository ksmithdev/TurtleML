﻿namespace TurtleML.Layers;

using System;
using System.IO;

/// <summary>
/// Represents a softmax output layer in a neural network, which normalizes the outputs to probabilities.
/// </summary>
public sealed class SoftMaxOutputLayer : ILayer
{
    private SoftMaxOutputLayer(IOutput input)
    {
        var inputs = input.Outputs;
        var inputSize = inputs.Length;

        Outputs = new Tensor(inputSize);
    }

    /// <inheritdoc/>
    public Tensor Outputs { get; private set; } = Tensor.Empty;

    /// <inheritdoc/>
    public Tensor Backpropagate(Tensor inputs, Tensor errors, float learningRate, float momentumRate)
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

    /// <inheritdoc/>
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
    /// Builder class for constructing <see cref="SoftMaxOutputLayer"/> instances.
    /// </summary>
    public class Builder : ILayerBuilder
    {
        /// <inheritdoc/>
        public ILayer Build(IOutput input)
        {
            return new SoftMaxOutputLayer(input);
        }
    }
}