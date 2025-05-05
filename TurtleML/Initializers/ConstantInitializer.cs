namespace TurtleML.Initializers;

using System;

/// <summary>
/// Initializes weights with a constant value.
/// </summary>
/// <remarks>
/// This initializer returns the same value for all weight samples.
/// </remarks>
public class ConstantInitializer : IInitializer
{
    private readonly float value;

    /// <summary>
    /// Initializes a new instance of the <see cref="ConstantInitializer"/> class.
    /// </summary>
    /// <param name="value">The constant value to use for initialization.</param>
    public ConstantInitializer(float value)
    {
        this.value = value;
    }

    /// <inheritdoc />
    public float Sample(int inputs, int outputs, Random random)
    {
        return value;
    }
}