using System;
using System.IO;

namespace TurtleML.Layers
{
    public class ConcurrentLayer : ILayer
    {
        private readonly ILayer inputLayer;
        private readonly ILayer[] layers;
        private readonly Tensor output;
        private readonly Tensor signals;

        private ConcurrentLayer(ILayerBuilder[] layers, ILayer inputLayer)
        {
            this.inputLayer = inputLayer;
            this.layers = new ILayer[layers.Length];

            int outputSize = 0;
            for (int l = 0; l < layers.Length; l++)
            {
                this.layers[l] = layers[l].Build(inputLayer);

                outputSize += this.layers[l].Outputs.Length;
            }

            output = new Tensor(outputSize);
            signals = new Tensor(inputLayer.Outputs.Dimensions);
        }

        public Tensor Outputs => output;

        public Tensor Backpropagate(Tensor errors, float learningRate)
        {
            signals.Clear();

            int signalSize = 0;
            for (int l = 0; l < layers.Length; l++)
            {
                var layer = layers[l];

                var error = new Tensor(layer.Outputs.Dimensions);
                error.Load(errors, signalSize);

                signals.Add(layer.Backpropagate(error, learningRate));

                signalSize += error.Length;
            }

            return signals;
        }

        public Tensor CalculateOutputs(Tensor inputs, bool training = false)
        {
            int outputSize = 0;
            for (int l = 0; l < layers.Length; l++)
            {
                var outputs = layers[l].CalculateOutputs(inputs, training);
                outputs.CopyTo(output, outputSize);

                outputSize += layers[l].Outputs.Length;
            }

            return output;
        }

        public void Dump(BinaryWriter writer)
        {
            for (int l = 0; l < layers.Length; l++)
                layers[l].Dump(writer);
        }

        public void Initialize(Random random)
        {
            for (int l = 0; l < layers.Length; l++)
                layers[l].Initialize(random);
        }

        public void Restore(BinaryReader reader)
        {
            for (int l = 0; l < layers.Length; l++)
                layers[l].Restore(reader);
        }

        public class Builder : ILayerBuilder
        {
            private ILayerBuilder[] layers;
            private int splitSize;

            public ILayer Build(ILayer inputLayer)
            {
                return new ConcurrentLayer(layers, inputLayer);
            }

            public Builder Layers(params ILayerBuilder[] layers)
            {
                this.layers = layers;

                return this;
            }
        }
    }
}