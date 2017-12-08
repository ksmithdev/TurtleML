namespace TurtleML
{
    public interface ILearningPolicy
    {
        float GetLearningRate(int epoch);
    }
}