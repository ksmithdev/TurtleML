using System;

namespace TurtleML.Loss
{
    public class CrossEntropyLoss : LossFunctionBase
    {
        public override float Calculate(float actual, float expected) => -expected * (float)Math.Log(actual);
    }
}