namespace TurtleML.LearningPolicies;

using System;

public class StepDecayLearningPolicy : ILearningPolicy
{
    private readonly float decay;
    private readonly float initialLearningRate;
    private readonly int step;

    public StepDecayLearningPolicy(float initialLearningRate, float decay, int step)
    {
        this.initialLearningRate = initialLearningRate;
        this.decay = decay;
        this.step = step;
    }

    public float GetLearningRate(int epoch)
    {
        return initialLearningRate * (float)Math.Pow(decay, (1 + epoch) / step);
    }
}