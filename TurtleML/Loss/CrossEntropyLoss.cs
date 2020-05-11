using System;

namespace TurtleML.Loss
{
    public class CrossEntropyLoss : ILossFunction
    {
        public float Calculate(float actual, float expected) => -expected * (float)Math.Log(actual);

        public float CalculateTotal(Tensor actuals, Tensor expected)
        {
            float sumCost = 0f;
            for (int i = 0, count = actuals.Length; i < count; i++)
            {
                sumCost += Calculate(actuals[i], expected[i]);
            }

            return sumCost;
        }

        public float Derivative(float actual, float expected) => actual - expected;
    }
}