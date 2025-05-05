namespace TurtleML.Activations;

using System.IO;

/// <summary>
/// Represents the Identity activation function, which simply returns the input value unchanged.
/// </summary>
public class IdentityActivation : IActivationFunction
{
    /// <inheritdoc />
    public float Activate(float value)
    {
        return value;
    }

    /// <inheritdoc />
    public float Derivative(float value)
    {
        return 1f;
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