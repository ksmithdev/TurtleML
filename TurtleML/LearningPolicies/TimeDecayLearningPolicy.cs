namespace TurtleML.LearningPolicies
{
    public class TimeDecayLearningPolicy : ILearningPolicy
    {
        private readonly float decay;
        private readonly float initialLearningRate;

        public TimeDecayLearningPolicy(float initialLearningRate, float decay)
        {
            this.initialLearningRate = initialLearningRate;
            this.decay = decay;
        }

        public float GetLearningRate(int epoch)
        {
            return initialLearningRate / (1f + (epoch / decay));
        }
    }
}