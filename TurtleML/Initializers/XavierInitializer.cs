namespace TurtleML.Initializers;

using System;

public class XavierInitializer : IInitializer
{
    public float Sample(int inputs, int outputs, Random random)
    {
        var max = 2f / (inputs + outputs);
        var min = -max;

        return ((float)random.NextDouble() * (max - min)) + min;
    }
}