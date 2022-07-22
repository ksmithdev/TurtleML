namespace TurtleML;

using System.IO;

/// <summary>
/// Represents an activation function and derivative for layers.
/// </summary>
public interface IActivationFunction
{
    /// <summary>
    /// Calculate the activation for the supplied value.
    /// </summary>
    /// <param name="value">The value to activate.</param>
    /// <returns>The activated value.</returns>
    float Activate(float value);

    /// <summary>
    /// Calculates the derivative for the supplied error signal.
    /// </summary>
    /// <param name="value">The signal value to derive.</param>
    /// <returns>The derivative value.</returns>
    float Derivative(float value);

    /// <summary>
    /// Dump the activation internals into the writer for saving.
    /// </summary>
    /// <param name="writer">The writer.</param>
    void Dump(BinaryWriter writer);

    /// <summary>
    /// Restore the activation internals from the reader.
    /// </summary>
    /// <param name="reader">The reader.</param>
    void Restore(BinaryReader reader);
}