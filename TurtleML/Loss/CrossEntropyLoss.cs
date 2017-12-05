using System;

namespace TurtleML.Loss
{
    public class CrossEntropyLoss : ILossFunction
    {
        public float CalculateCost(Tensor actuals, Tensor expected)
        {
            float sumCost = 0f;
            for (int i = 0, count = actuals.Length; i < count; i++)
                sumCost += -expected[i] * (float)Math.Log(actuals[i]) - (1 - expected[i]) * (float)Math.Log(1 - actuals[i]);
            return sumCost;
        }
    }
}