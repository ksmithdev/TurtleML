namespace TurtleML;

using System;
using System.IO;

/// <summary>
/// Represents an individual layer in a deep learning model.
/// </summary>
public interface ILayer : IOutput
{
    /// <summary>
    /// 
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="errors"></param>
    /// <param name="learningRate"></param>
    /// <param name="momentumRate"></param>
    /// <returns></returns>
    Tensor Backpropagate(Tensor inputs, Tensor errors, float learningRate, float momentumRate);

    Tensor CalculateOutputs(Tensor inputs, bool training = false);

    void Dump(BinaryWriter writer);

    void Initialize(Random random);

    void Restore(BinaryReader reader);
}