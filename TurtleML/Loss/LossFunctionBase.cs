namespace TurtleML.Loss
{
    public abstract class LossFunctionBase : ILossFunction
    {
        public abstract float Calculate(float actual, float expected);

        public float CalculateTotal(Tensor actuals, Tensor expected)
        {
            float sumErrorCost = 0f;
            for (int o = 0; o < actuals.Length; o++)
            {
                sumErrorCost += Calculate(actuals[o], expected[o]);
            }

            return sumErrorCost;
        }
    }
}