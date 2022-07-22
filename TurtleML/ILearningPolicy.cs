namespace TurtleML;

/// <summary>
/// Represents the learning policy for the training network.
/// </summary>
public interface ILearningPolicy
{
    /// <summary>
    /// Returns the learning rate for the supplied epoch.
    /// </summary>
    /// <param name="epoch">The learning epoch.</param>
    /// <returns>The learning rate.</returns>
    float GetLearningRate(int epoch);
}