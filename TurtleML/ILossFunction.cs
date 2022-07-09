namespace TurtleML
{
    public interface ILossFunction
    {
        float Calculate(float actual, float expected);

        float CalculateTotal(Tensor actuals, Tensor expected);
    }
}