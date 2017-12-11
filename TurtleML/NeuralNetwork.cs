using System;
using System.Diagnostics;
using System.IO;
using TurtleML.Loss;

namespace TurtleML
{
    public class NeuralNetwork
    {
        private readonly ILayer[] layers;
        private readonly ILearningPolicy learningPolicy;
        private readonly ILossFunction loss;
        private readonly float momentum;
        private readonly Random seed;
        private readonly bool shuffle;

        private NeuralNetwork(float momentum, bool shuffle, Random seed, ILearningPolicy learningPolicy, ILossFunction loss, ILayerBuilder[] layers)
        {
            this.momentum = momentum;
            this.shuffle = shuffle;
            this.seed = seed;
            this.learningPolicy = learningPolicy;
            this.loss = loss;
            this.layers = new ILayer[layers.Length];

            ILayer inputLayer = null;
            for (int l = 0; l < layers.Length; l++)
            {
                inputLayer = layers[l].Build(inputLayer);
                inputLayer.Initialize(seed);

                this.layers[l] = inputLayer;
            }
        }

        public event EventHandler<TrainingProgressEventArgs> TrainingProgress;

        public ILayer[] Layers => layers;

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

        public void Train(TrainingSet trainingSet, TrainingSet validationSet, int epochs)
        {
            var stopwatch = new Stopwatch();
            for (int i = 0; i < epochs; i++)
            {
                float learningRate = learningPolicy.GetLearningRate(i);

                stopwatch.Restart();
                float trainingError = Train(trainingSet, learningRate);
                float validationError = Test(validationSet);
                stopwatch.Stop();

                RaiseTrainingProgress(trainingError, validationError, learningRate, stopwatch.ElapsedMilliseconds);
            }
        }

        protected void RaiseTrainingProgress(float trainingError, float validationError, float learningRate, long cycleTime) => TrainingProgress?.Invoke(this, new TrainingProgressEventArgs(trainingError, validationError, learningRate, cycleTime));

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
            private ILearningPolicy learning;
            private ILearningPolicy learningPolicy;
            private float learningRate;
            private ILossFunction loss = new MeanSquaredError();
            private float momentum;
            private Random seed;
            private bool shuffle = false;

            public NeuralNetwork Build()
            {
                return new NeuralNetwork(momentum, shuffle, seed, learningPolicy, loss, layers);
            }

            public Builder Layers(params ILayerBuilder[] layers)
            {
                this.layers = layers;

                return this;
            }

            public Builder LearningPolicy(ILearningPolicy learningPolicy, float momentum)
            {
                this.learningPolicy = learningPolicy;

                return this;
            }

            [Obsolete("Obsolete. Use LearningPolicy() instead.")]
            public Builder LearningRate(float learningRate, float momentum)
            {
                this.learningRate = learningRate;
                this.momentum = momentum;

                return this;
            }

            public Builder Loss(ILossFunction loss)
            {
                this.loss = loss;

                return this;
            }

            public Builder Seed(Random seed)
            {
                this.seed = seed;

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