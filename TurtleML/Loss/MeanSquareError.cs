namespace TurtleML.Loss;

public class MeanSquareError : LossFunctionBase
{
    public override float Calculate(float actual, float expected) => 0.5f * (actual - expected) * (actual - expected);
}