namespace TurtleML
{
    public interface ILossFunction
    {
        float CalculateCost(Tensor actuals, Tensor expected);
    }
}