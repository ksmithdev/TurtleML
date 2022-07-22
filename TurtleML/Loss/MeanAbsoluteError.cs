namespace TurtleML.Loss;

using System;

public class MeanAbsoluteError : LossFunctionBase
{
    public override float Calculate(float actual, float expected) => Math.Abs(actual - expected);
}