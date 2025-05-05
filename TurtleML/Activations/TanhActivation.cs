namespace TurtleML.Activations;

using System;
using System.IO;

/// <summary>
/// Hyperbolic tangent activation function. Maps inputs to the range [-1, 1] with a smooth gradient.
/// </summary>
public class TanhActivation : IActivationFunction
{
    /// <inheritdoc />
    public float Activate(float value)
    {
        return (float)Math.Tanh(value);
    }

    /// <inheritdoc />
    public float Derivative(float value)
    {
        return 1f - (value * value);
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