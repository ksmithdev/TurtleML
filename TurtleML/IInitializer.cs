namespace TurtleML;

using System;

/// <summary>
/// Represents a method for initializing weights in a layer.
/// </summary>
public interface IInitializer
{
    /// <summary>
    /// Returns the initial value for a weight based on the number of inputs, outputs, and a seed randomizer.
    /// </summary>
    /// <param name="inputs">The number of inputs for the weight.</param>
    /// <param name="outputs">The number of outputs for the weight.</param>
    /// <param name="random">A seeded randomizer.</param>
    /// <returns>The initial value for a weight.</returns>
    float Sample(int inputs, int outputs, Random random);
}