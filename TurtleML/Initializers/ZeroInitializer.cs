namespace TurtleML.Initializers;

using System;

public class ZeroInitializer : IInitializer
{
    public float Sample(int inputs, int outputs, Random random)
    {
        return 0f;
    }
}