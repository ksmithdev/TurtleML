using System;

namespace TurtleML.Activations
{
    public class SigmoidActivation : IActivationFunction
    {
        public float Activate(float value)
        {
            return 1f / (1f + (float)Math.Exp(-value));
        }

        public float Derivative(float value)
        {
            return value * (1f - value);
        }
    }
}