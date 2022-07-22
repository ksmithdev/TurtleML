namespace TurtleML;

/// <summary>
/// Represents an interface for output values.
/// </summary>
public interface IOutput
{
    /// <summary>
    /// Gets the output values.
    /// </summary>
    Tensor Outputs { get; }
}