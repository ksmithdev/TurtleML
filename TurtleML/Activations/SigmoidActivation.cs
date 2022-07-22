namespace TurtleML.Activations;

using System;
using System.IO;

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

    public void Dump(BinaryWriter writer)
    {
    }

    public void Restore(BinaryReader reader)
    {
    }
}