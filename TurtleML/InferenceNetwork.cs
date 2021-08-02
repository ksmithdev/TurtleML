using System;
using System.IO;

namespace TurtleML
{
    public class InferenceNetwork
    {
        protected readonly ILayer[] Layers;
        protected readonly ILossFunction Loss;

        protected InferenceNetwork(ILossFunction loss, ILayer[] layers)
        {
            Loss = loss;
            Layers = layers;
        }

        public static InferenceNetwork Restore(string fileName)
        {
            using (var file = File.OpenRead(fileName))
            {
                return Restore(file);
            }
        }

        public static InferenceNetwork Restore(Stream stream)
        {
            using (var reader = new BinaryReader(stream))
            {
                var magicNumber = reader.ReadChars(3);
                if (magicNumber != new[] { 't', 'n', 'n' })
                {
                    throw new InvalidDataException("Source data is not a valid neural network dump.");
                }

                var version = reader.ReadInt32();
                if (version != 1)
                {
                    throw new InvalidOperationException($"Cannot read version {version} format.");
                }

                var lossFunctionType = reader.ReadString();
                if (!(Activator.CreateInstance(Type.GetType(lossFunctionType)) is ILossFunction lossFunction))
                {
                    throw new InvalidOperationException($"An invalid loss function \"{lossFunctionType}\" was specified.");
                }

                var layerCount = reader.ReadInt32();
                var layerTypes = new string[layerCount];
                for (int i = 0; i < layerCount; i++)
                {
                    layerTypes[i] = reader.ReadString();
                }

                var layers = new ILayer[layerCount];
                for (int i = 0; i < layerCount; i++)
                {
                    var layerType = reader.ReadString();
                    if (!(Activator.CreateInstance(Type.GetType(layerType)) is ILayer layer))
                    {
                        throw new InvalidOperationException($"An invalid layer \"{layerType}\" was specified.");
                    }
                    layers[i] = layer;
                }

                foreach (var layer in layers)
                {
                    layer.Restore(reader);
                }

                return new InferenceNetwork(lossFunction, layers);
            }
        }

        public Tensor CalculateOutputs(Tensor inputs)
        {
            var results = inputs;
            for (int l = 0, count = Layers.Length; l < count; l++)
            {
                results = Layers[l].CalculateOutputs(results, training: false);
            }

            return results;
        }
    }
}