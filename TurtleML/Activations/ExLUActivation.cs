namespace TurtleML.Activations;

using System;
using System.IO;

/// <summary>
/// Exponential Linear Unit (ExLU) activation function.
/// </summary>
/// <remarks>
/// For x &gt;= 0: f(x) = x.
/// For x &lt; 0: f(x) = alpha * (e^x - 1).
/// </remarks>
public class ExLUActivation : IActivationFunction
{
    private float alpha;

    /// <summary>
    /// Initializes a new instance of the <see cref="ExLUActivation"/> class.
    /// </summary>
    /// <param name="alpha">The exponential scaling factor for negative values (default: 1.0f).</param>
    public ExLUActivation(float alpha = 1f)
    {
        this.alpha = alpha;
    }

    /// <inheritdoc />
    public float Activate(float value)
    {
        return value < 0f ? alpha * ((float)Math.Exp(value) - 1f) : value;
    }

    /// <inheritdoc />
    public float Derivative(float value)
    {
        return value < 0f ? value + alpha : 1f;
    }

    /// <inheritdoc />
    public void Dump(BinaryWriter writer)
    {
        writer.Write(alpha);
    }

    /// <inheritdoc />
    public void Restore(BinaryReader reader)
    {
        alpha = reader.ReadSingle();
    }
}