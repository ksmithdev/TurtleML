namespace TurtleML.Activations;

using System;
using System.IO;

/// <summary>
/// Sigmoid activation function with logistic curve behavior.
/// </summary>
public class SigmoidActivation : IActivationFunction
{
    /// <inheritdoc />
    public float Activate(float value)
    {
        return 1f / (1f + (float)Math.Exp(-value));
    }

    /// <inheritdoc />
    public float Derivative(float value)
    {
        return value * (1f - value);
    }

    /// <inheritdoc />
    public void Dump(BinaryWriter writer)
    {
    }

    /// <inheritdoc />
    public void Restore(BinaryReader reader)
    {
    }
}