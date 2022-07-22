namespace TurtleML;

/// <summary>
/// Represents an interface for building a layer.
/// </summary>
public interface ILayerBuilder
{
    /// <summary>
    /// Returns the constructed layer.
    /// </summary>
    /// <param name="input">The input information for the layer.</param>
    /// <returns>A layer.</returns>
    ILayer Build(IOutput input);
}