namespace TurtleML.LearningPolicies;

/// <summary>
/// A learning rate policy that uses a fixed (constant) learning rate throughout all epochs.
/// </summary>
/// <remarks>
/// This policy does not adapt the learning rate over time. The learning rate remains
/// unchanged regardless of the training epoch.
/// </remarks>
public class FixedLearningPolicy : ILearningPolicy
{
    private readonly float learningRate;

    /// <summary>
    /// Initializes a new instance of the <see cref="FixedLearningPolicy"/> class.
    /// </summary>
    /// <param name="learningRate">The fixed learning rate to use for all epochs.</param>
    public FixedLearningPolicy(float learningRate)
    {
        this.learningRate = learningRate;
    }

    /// <inheritdoc/>
    public float GetLearningRate(int epoch)
    {
        return learningRate;
    }
}