using System;
using System.Collections.Generic;
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

        public IEnumerable<ILayer> InternalLayers => Layers;

        public static InferenceNetwork Restore(FileInfo fileInfo)
        {
            using (var file = fileInfo.OpenRead())
            {
                return Restore(file);
            }
        }

        public static InferenceNetwork Restore(Stream stream)
        {
            using (var reader = new BinaryReader(stream))
            {
                var magicNumber = reader.ReadChars(3);
                if (magicNumber[0] != 't' || magicNumber[1] != 'n' || magicNumber[2] != 'n')
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
                    if (!(Activator.CreateInstance(Type.GetType(layerTypes[i])) is ILayer layer))
                    {
                        throw new InvalidOperationException($"An invalid layer \"{layerTypes[i]}\" was specified.");
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