namespace TurtleML.Initializers;

using System;

/// <summary>
/// Initializes weights using He initialization (also known as Kaiming initialization).
/// </summary>
/// <remarks>
/// This method is recommended for layers with ReLU activation functions.
/// The values are sampled from a uniform distribution in the range [-sqrt(2/(input_size)), sqrt(2/(input_size))].
/// </remarks>
public class HeInitializer : IInitializer
{
    /// <inheritdoc />
    public float Sample(int inputs, int outputs, Random random)
    {
        var max = 2f / inputs;
        var min = -max;

        return ((float)random.NextDouble() * (max - min)) + min;
    }
}