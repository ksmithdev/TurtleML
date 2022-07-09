namespace TurtleML.Tests
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    /// <summary>
    /// Defines a layer used for stubbing input and output.
    /// </summary>
    class StubLayer : ILayer
    {
        public Tensor Outputs { get; set; }

        public Tensor Backpropagate(Tensor errors, float learningRate, float momentumRate)
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
}
