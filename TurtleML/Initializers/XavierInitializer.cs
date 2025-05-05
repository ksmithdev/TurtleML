namespace TurtleML.Initializers;

using System;

/// <summary>
/// Initializes weights using Xavier initialization (also known as Glorot initialization).
/// </summary>
/// <remarks>
/// This method is recommended for layers with sigmoid or tanh activation functions.
/// The values are sampled from a uniform distribution in the range [-sqrt(2/(input_size + output_size)), sqrt(2/(input_size + output_size))].
/// </remarks>
public class XavierInitializer : IInitializer
{
    /// <inheritdoc />
    public float Sample(int inputs, int outputs, Random random)
    {
        var max = 2f / (inputs + outputs);
        var min = -max;

        return ((float)random.NextDouble() * (max - min)) + min;
    }
}