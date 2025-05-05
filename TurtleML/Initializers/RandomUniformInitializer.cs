namespace TurtleML.Initializers;

using System;

/// <summary>
/// Initializes weights using a uniform distribution between specified bounds.
/// </summary>
/// <remarks>
/// This initializer samples values from a uniform distribution [min, max).
/// </remarks>
public class RandomUniformInitializer : IInitializer
{
    private readonly float max;
    private readonly float min;

    /// <summary>
    /// Initializes a new instance of the <see cref="RandomUniformInitializer"/> class.
    /// </summary>
    /// <param name="min">The lower bound of the uniform distribution.</param>
    /// <param name="max">The upper bound of the uniform distribution.</param>
    public RandomUniformInitializer(float min, float max)
    {
        this.min = min;
        this.max = max;
    }

    /// <inheritdoc />
    public float Sample(int inputs, int outputs, Random random)
    {
        return ((float)random.NextDouble() * (max - min)) + min;
    }
}