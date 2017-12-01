namespace TurtleML.Activations
{
    public class IdentityActivation : IActivationFunction
    {
        public float Activate(float value)
        {
            return value;
        }

        public float Derivative(float value)
        {
            return 1f;
        }
    }
}