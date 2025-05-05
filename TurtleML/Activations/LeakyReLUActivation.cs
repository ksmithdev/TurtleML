namespace TurtleML.Activations;

using System.IO;

/// <summary>
/// Leaky ReLU (Rectified Linear Unit) activation function. Allows small negative values to pass through with a configurable slope.
/// </summary>
public class LeakyReLUActivation : IActivationFunction
{
    private float leak;

    /// <summary>
    /// Initializes a new instance of the <see cref="LeakyReLUActivation"/> class.
    /// </summary>
    /// <param name="leak">Small negative slope for x ≤ 0 (default: 0.01f).</param>
    public LeakyReLUActivation(float leak = 0.01f)
    {
        this.leak = leak;
    }

    /// <inheritdoc />
    public float Activate(float value)
    {
        return value > 0f ? value : leak * value;
    }

    /// <inheritdoc />
    public float Derivative(float value)
    {
        return value >= 0f ? 1f : leak;
    }

    /// <inheritdoc />
    public void Dump(BinaryWriter writer)
    {
        writer.Write(leak);
    }

    /// <inheritdoc />
    public void Restore(BinaryReader reader)
    {
        leak = reader.ReadSingle();
    }
}