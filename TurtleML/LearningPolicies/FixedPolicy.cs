namespace TurtleML.LearningPolicies
{
    public class FixedPolicy : ILearningPolicy
    {
        private readonly float learningRate;

        public FixedPolicy(float learningRate)
        {
            this.learningRate = learningRate;
        }

        public float GetLearningRate(int epoch)
        {
            return learningRate;
        }
    }
}