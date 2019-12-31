using System;

namespace TurtleML.Loss
{
    public class MeanAbsoluteError : ILossFunction
    {
        public float Calculate(float actual, float expected) => Math.Abs(actual - expected);

        public float CalculateTotal(Tensor actuals, Tensor expected)
        {
            float absErrorCost = 0f;
            for (int o = 0; o < actuals.Length; o++)
                absErrorCost += Calculate(actuals[o], expected[o]);
            return absErrorCost;
        }

        public float Derivative(float actual, float expected) => expected == actual ? 0f : expected > actual ? 1f : -1f;
    }
}