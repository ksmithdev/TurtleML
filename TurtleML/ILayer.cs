using System.IO;

namespace TurtleML
{
    public interface ILayer
    {
        Tensor Outputs { get; }

        void Backpropagate(Tensor errors, float learningRate);

        Tensor CalculateOutputs(Tensor inputs, bool training = false);

        void Dump(BinaryWriter writer);

        void Restore(BinaryReader reader);
    }
}