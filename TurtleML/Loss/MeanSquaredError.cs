namespace TurtleML.Loss
{
    public class MeanSquaredError : ILossFunction
    {
        public float CalculateCost(Tensor actuals, Tensor expected)
        {
            float sumErrorCost = 0f;
            for (int o = 0; o < actuals.Length; o++)
                sumErrorCost += 0.5f * (expected[o] - actuals[o]) * (expected[o] - actuals[o]);
            return sumErrorCost;
        }
    }
}