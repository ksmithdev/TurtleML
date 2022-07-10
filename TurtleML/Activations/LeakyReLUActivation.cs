namespace TurtleML.Activations
{
    using System.IO;

    public class LeakyReLUActivation : IActivationFunction
    {
        private float leak;

        public LeakyReLUActivation(float leak = 0.01f)
        {
            this.leak = leak;
        }

        public float Activate(float value)
        {
            return value > 0f ? value : leak * value;
        }

        public float Derivative(float value)
        {
            return value >= 0f ? 1f : leak;
        }

        public void Dump(BinaryWriter writer)
        {
            writer.Write(leak);
        }

        public void Restore(BinaryReader reader)
        {
            leak = reader.ReadSingle();
        }
    }
}