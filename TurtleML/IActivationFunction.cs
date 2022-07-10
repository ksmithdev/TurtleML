namespace TurtleML
{
    using System.IO;

    public interface IActivationFunction
    {
        float Activate(float value);

        float Derivative(float value);
        void Dump(BinaryWriter writer);
        void Restore(BinaryReader reader);
    }
}