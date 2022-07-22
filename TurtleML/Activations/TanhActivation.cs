namespace TurtleML.Activations;

using System;
using System.IO;

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

    public void Dump(BinaryWriter writer)
    {
    }

    public void Restore(BinaryReader reader)
    {
    }
}