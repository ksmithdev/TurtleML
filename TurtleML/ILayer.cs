using System;
using System.IO;

namespace TurtleML
{
    public interface ILayer : IOutput
    {
        Tensor Backpropagate(Tensor errors, float learningRate, float momentumRate);

        Tensor CalculateOutputs(Tensor inputs, bool training = false);

        void Dump(BinaryWriter writer);

        void Initialize(Random random);

        void Restore(BinaryReader reader);
    }
}