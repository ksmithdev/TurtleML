using System;

namespace TurtleML.Activations
{
    public class ExLUActivation : IActivationFunction
    {
        private readonly float alpha;

        public ExLUActivation(float alpha = 1f)
        {
            this.alpha = alpha;
        }

        public float Activate(float value)
        {
            return value < 0f ? alpha * ((float)Math.Exp(value) - 1f) : value;
        }

        public float Derivative(float value)
        {
            return value < 0f ? value + alpha : 1f;
        }
    }
}