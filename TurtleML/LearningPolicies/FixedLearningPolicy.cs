namespace TurtleML.LearningPolicies;

public class FixedLearningPolicy : ILearningPolicy
{
    private readonly float learningRate;

    public FixedLearningPolicy(float learningRate)
    {
        this.learningRate = learningRate;
    }

    public float GetLearningRate(int epoch)
    {
        return learningRate;
    }
}