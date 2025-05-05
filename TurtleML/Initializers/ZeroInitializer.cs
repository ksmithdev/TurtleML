namespace TurtleML.Initializers;

using System;

/// <summary>
/// Initializes weights with zero values.
/// </summary>
/// <remarks>
/// This initializer sets all weights to exactly 0.0f.
/// </remarks>
public class ZeroInitializer : IInitializer
{
    /// <inheritdoc />
    public float Sample(int inputs, int outputs, Random random)
    {
        return 0f;
    }
}