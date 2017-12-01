namespace TurtleML
{
    public interface IActivationFunction
    {
        float Activate(float value);

        float Derivative(float value);
    }
}