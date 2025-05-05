namespace TurtleML.LearningPolicies;

/// <summary>
/// A learning rate policy that decays the learning rate continuously over time.
/// </summary>
/// <remarks>
/// The learning rate decreases according to a hyperbolic schedule.
/// Formula: learning_rate = initial_learning_rate / (1 + (epoch / decay)).
/// </remarks>
public class TimeDecayLearningPolicy : ILearningPolicy
{
    private readonly float decay;
    private readonly float initialLearningRate;

    /// <summary>
    /// Initializes a new instance of the <see cref="TimeDecayLearningPolicy"/> class.
    /// </summary>
    /// <param name="initialLearningRate">The starting learning rate.</param>
    /// <param name="decay">The time constant that controls the decay speed (larger values = slower decay).</param>
    public TimeDecayLearningPolicy(float initialLearningRate, float decay)
    {
        this.initialLearningRate = initialLearningRate;
        this.decay = decay;
    }

    /// <inheritdoc/>
    public float GetLearningRate(int epoch)
    {
        return initialLearningRate / (1f + (epoch / decay));
    }
}