namespace TurtleML.Loss;

using System;

public class CrossEntropyLoss : LossFunctionBase
{
    public override float Calculate(float actual, float expected) => -expected * (float)Math.Log(actual);
}