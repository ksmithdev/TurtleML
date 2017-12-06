using System;
using System.IO;
using TurtleML.Loss;

namespace TurtleML
{
    public class NeuralNetwork
    {
        private readonly ILayer[] layers;
        private readonly ILossFunction loss;
        private readonly Random random;
        private readonly bool shuffle;

        private NeuralNetwork(bool shuffle, Random random, ILossFunction loss, ILayerBuilder[] layers)
        {
            this.shuffle = shuffle;
            this.random = random;
            this.loss = loss;
            this.layers = new ILayer[layers.Length];

            ILayer inputLayer = null;
            for (int l = 0; l < layers.Length; l++)
            {
                inputLayer = layers[l].Build(inputLayer);
                inputLayer.Initialize(random);

                this.layers[l] = inputLayer;
            }
        }

        public Tensor CalculateOutputs(Tensor inputs)
        {
            Tensor results = inputs;
            for (int l = 0, count = layers.Length; l < count; l++)
                results = layers[l].CalculateOutputs(results, training: false);
            return results;
        }

        public void Dump(string path)
        {
            using (var file = File.Create(path))
                Dump(file);
        }

        public void Dump(Stream stream)
        {
            using (var writer = new BinaryWriter(stream))
                foreach (var layer in layers)
                    layer.Dump(writer);
        }

        public void Restore(string fileName)
        {
            using (var file = File.OpenRead(fileName))
                Restore(file);
        }

        public void Restore(Stream stream)
        {
            using (var reader = new BinaryReader(stream))
                foreach (var layer in layers)
                    layer.Restore(reader);
        }

        public float Test(TrainingSet trainingSet)
        {
            float sumCost = 0f;
            for (int i = 0, count = trainingSet.Count; i < count; i++)
            {
                var inputs = trainingSet[i].Item1;
                var expected = trainingSet[i].Item2;

                Tensor actuals = CalculateOutputs(inputs);
                Tensor errors = new Tensor(actuals.Length);

                for (int o = 0; o < actuals.Length; o++)
                    errors[o] = expected[o] - actuals[o];

                sumCost += loss.CalculateCost(actuals, expected);
            }

            return sumCost / trainingSet.Count;
        }

        public float Train(TrainingSet trainingSet, float learningRate)
        {
            if (shuffle)
                trainingSet.Shuffle();

            float sumCost = 0f;
            for (int i = 0, count = trainingSet.Count; i < count; i++)
            {
                var inputs = trainingSet[i].Item1;
                var expected = trainingSet[i].Item2;

                Tensor actuals = CalculateTrainingOutputs(inputs);
                Tensor errors = new Tensor(actuals.Length);

                for (int o = 0; o < actuals.Length; o++)
                    errors[o] = expected[o] - actuals[o];

                sumCost += loss.CalculateCost(actuals, expected);

                BackPropagate(errors, learningRate);
            }

            return sumCost / trainingSet.Count;
        }

        private Tensor BackPropagate(Tensor errors, float learningRate)
        {
            Tensor signals = errors;
            for (int l = layers.Length - 1; l > -1; l--)
                signals = layers[l].Backpropagate(signals, learningRate);
            return signals;
        }

        private Tensor CalculateTrainingOutputs(Tensor inputs)
        {
            Tensor results = inputs;
            for (int l = 0, count = layers.Length; l < count; l++)
                results = layers[l].CalculateOutputs(results, training: true);
            return results;
        }

        public class Builder
        {
            private ILayerBuilder[] layers;
            private ILossFunction loss = new MeanSquaredError();
            private Random random;
            private bool shuffle = false;

            public NeuralNetwork Build()
            {
                return new NeuralNetwork(shuffle, random, loss, layers);
            }

            public Builder Layers(params ILayerBuilder[] layers)
            {
                this.layers = layers;

                return this;
            }

            public Builder Loss(ILossFunction loss)
            {
                this.loss = loss;

                return this;
            }

            public Builder Seed(Random random)
            {
                this.random = random;

                return this;
            }

            public Builder Shuffle(bool shuffle)
            {
                this.shuffle = shuffle;

                return this;
            }
        }
    }
}