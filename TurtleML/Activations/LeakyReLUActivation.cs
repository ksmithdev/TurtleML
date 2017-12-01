namespace TurtleML.Activations
{
    public class LeakyReLUActivation : IActivationFunction
    {
        private readonly float leak;

        public LeakyReLUActivation(float leak = 0.01f)
        {
            this.leak = leak;
        }

        public float Activate(float value)
        {
            return value > 0f ? value : leak * value;
        }

        public float Derivative(float value)
        {
            return value >= 0f ? 1f : leak;
        }
    }
}