namespace TurtleML.Loss
{
    public class MeanSquareError : ILossFunction
    {
        public float Calculate(float actual, float expected) => 0.5f * (actual - expected) * (actual - expected);

        public float CalculateTotal(Tensor actuals, Tensor expected)
        {
            float sumErrorCost = 0f;
            for (int o = 0; o < actuals.Length; o++)
            {
                sumErrorCost += Calculate(actuals[o], expected[o]);
            }

            return sumErrorCost;
        }

        public float Derivative(float actual, float expected) => -(expected - actual);
    }
}