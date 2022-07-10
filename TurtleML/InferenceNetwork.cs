using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;

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
            using var file = fileInfo.OpenRead();
            return Restore(file);
        }

        public static InferenceNetwork Restore(Stream stream)
        {
            using var reader = new BinaryReader(stream);
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
            var layers = new ILayer[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                var layerType = reader.ReadString();
                if (RuntimeHelpers.GetUninitializedObject(Type.GetType(layerType)) is not ILayer layer)
                {
                    throw new InvalidOperationException($"An invalid layer \"{layerType}\" was specified.");
                }
                layer.Restore(reader);
                layers[i] = layer;
            }

            return new InferenceNetwork(lossFunction, layers);
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