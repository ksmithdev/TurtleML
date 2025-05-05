namespace TurtleML.LearningPolicies;

using System;

/// <summary>
/// A learning rate policy that decays the learning rate in discrete steps.
/// </summary>
/// <remarks>
/// The learning rate is multiplied by a decay factor at regular intervals (steps).
/// Formula: learning_rate = initial_learning_rate * (decay^((1 + epoch) / step)).
/// </remarks>
public class StepDecayLearningPolicy : ILearningPolicy
{
    private readonly float decay;
    private readonly float initialLearningRate;
    private readonly int step;

    /// <summary>
    /// Initializes a new instance of the <see cref="StepDecayLearningPolicy"/> class.
    /// </summary>
    /// <param name="initialLearningRate">The starting learning rate.</param>
    /// <param name="decay">The decay factor applied at each step (0 &lt; decay &lt; 1).</param>
    /// <param name="step">The number of epochs between consecutive decays.</param>
    public StepDecayLearningPolicy(float initialLearningRate, float decay, int step)
    {
        this.initialLearningRate = initialLearningRate;
        this.decay = decay;
        this.step = step;
    }

    /// <inheritdoc/>
    public float GetLearningRate(int epoch)
    {
        return initialLearningRate * (float)Math.Pow(decay, (1 + epoch) / step);
    }
}