using System;

namespace TurtleML.Loss
{
    public class MeanAbsoluteError : LossFunctionBase
    {
        public override float Calculate(float actual, float expected) => Math.Abs(actual - expected);
    }
}