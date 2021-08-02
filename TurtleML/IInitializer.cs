using System;

namespace TurtleML
{
    public interface IInitializer
    {
        float Sample(int inputs, int outputs, Random random);
    }
}