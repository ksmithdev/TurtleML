using System;

namespace TurtleML.Activations
{
    public class SoftPlusActivation : IActivationFunction
    {
        public float Activate(float value)
        {
            return (float)Math.Log(1.0 + Math.Exp(value));
        }

        public float Derivative(float value)
        {
            return 1f / (1f + (float)Math.Exp(-value));
        }
    }
}