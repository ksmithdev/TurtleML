namespace TurtleML.Activations;

using System;
using System.IO;

/// <summary>
/// Soft Plus activation function. Smooth approximation of ReLU that remains differentiable everywhere.
/// </summary>
public class SoftPlusActivation : IActivationFunction
{
    /// <inheritdoc />
    public float Activate(float value)
    {
        return (float)Math.Log(1.0 + Math.Exp(value));
    }

    /// <inheritdoc />
    public float Derivative(float value)
    {
        return 1f / (1f + (float)Math.Exp(-value));
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