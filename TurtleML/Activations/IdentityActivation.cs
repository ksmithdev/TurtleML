namespace TurtleML.Activations
{
    using System.IO;

    public class IdentityActivation : IActivationFunction
    {
        public float Activate(float value)
        {
            return value;
        }

        public float Derivative(float value)
        {
            return 1f;
        }

        public void Dump(BinaryWriter writer)
        {
        }

        public void Restore(BinaryReader reader)
        {
        }
    }
}