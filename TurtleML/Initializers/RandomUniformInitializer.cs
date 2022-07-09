using System;

namespace TurtleML.Initializers
{
    public class RandomUniformInitializer : IInitializer
    {
        private readonly float max;
        private readonly float min;

        public RandomUniformInitializer(float min, float max)
        {
            this.min = min;
            this.max = max;
        }

        public float Sample(int inputs, int outputs, Random random)
        {
            return ((float)random.NextDouble() * (max - min)) + min;
        }
    }
}