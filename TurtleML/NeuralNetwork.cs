using System;
using System.IO;

namespace TurtleML
{
    public class NeuralNetwork
    {
        protected readonly ILayer[] Layers;
        protected readonly ILossFunction Loss;

        protected NeuralNetwork(ILossFunction loss, ILayer[] layers)
        {
            Loss = loss;
            Layers = layers;
        }

        public static NeuralNetwork Restore(string fileName)
        {
            using (var file = File.OpenRead(fileName))
            {
                return Restore(file);
            }
        }

        public static NeuralNetwork Restore(Stream stream)
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
                var lossFunction = (ILossFunction)Activator.CreateInstance(Type.GetType(lossFunctionType));

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
                    layers[i] = (ILayer)Activator.CreateInstance(Type.GetType(layerType));
                }

                foreach (var layer in layers)
                {
                    layer.Restore(reader);
                }

                return new NeuralNetwork(lossFunction, layers);
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

        public void Dump(string path)
        {
            using (var file = File.Create(path))
            {
                Dump(file);
            }
        }

        public void Dump(Stream stream)
        {
            using (var writer = new BinaryWriter(stream))
            {
                // magic numbers
                writer.Write(new char[] { 't', 'n', 'n' });
                // file version
                writer.Write(1);
                // loss function
                writer.Write(Loss.GetType().AssemblyQualifiedName);
                // number of layers
                writer.Write(Layers.Length);
                foreach (var layer in Layers)
                {
                    writer.Write(layer.GetType().AssemblyQualifiedName);
                }
                // layer data dump
                foreach (var layer in Layers)
                {
                    layer.Dump(writer);
                }
            }
        }
    }
}