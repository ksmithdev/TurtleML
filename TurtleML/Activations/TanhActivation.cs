using System;

namespace TurtleML.Activations
{
    public class TanhActivation : IActivationFunction
    {
        public float Activate(float value)
        {
            return (float)Math.Tanh(value);
        }

        public float Derivative(float value)
        {
            return 1 - (value * value);
        }
    }
}