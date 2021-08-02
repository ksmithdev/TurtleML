using System;

namespace TurtleML.Initializers
{
    public class ConstantInitializer : IInitializer
    {
        private readonly float value;

        public ConstantInitializer(float value)
        {
            this.value = value;
        }

        public float Sample(int inputs, int outputs, Random random)
        {
            return value;
        }
    }
}