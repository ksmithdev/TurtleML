namespace TurtleML.Tests;

using System;
using System.IO;

/// <summary>
/// Defines a layer used for stubbing input and output.
/// </summary>
class StubLayer : ILayer
{
    public Tensor Outputs { get; set; }

    public Tensor Backpropagate(Tensor inputs, Tensor errors, float learningRate, float momentumRate)
    {
        return errors;
    }

    public Tensor CalculateOutputs(Tensor inputs, bool training = false)
    {
        Outputs = inputs;
        return inputs;
    }

    public void Dump(BinaryWriter writer)
    {
    }

    public void Initialize(Random random)
    {
    }

    public void Restore(BinaryReader reader)
    {
    }
}
