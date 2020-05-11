using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using TurtleML.Loss;

namespace TurtleML
{
    public sealed class NeuralNetwork
    {
        private readonly ILearningPolicy learningPolicy;
        private readonly ILossFunction loss;
        private readonly float momentumRate;
        private readonly bool shuffle;
        private bool aborted;

        private NeuralNetwork(float momentumRate, bool shuffle, Random seed, ILearningPolicy learningPolicy, ILossFunction loss, IReadOnlyList<ILayerBuilder> layers)
        {
            this.momentumRate = momentumRate;
            this.shuffle = shuffle;
            this.learningPolicy = learningPolicy;
            this.loss = loss;

            var layerCollection = new List<ILayer>(layers.Count);

            ILayer inputLayer = null;
            for (int l = 0; l < layers.Count; l++)
            {
                inputLayer = layers[l].Build(inputLayer);
                inputLayer.Initialize(seed);

                layerCollection.Insert(l, inputLayer);
            }

            Layers = new ReadOnlyCollection<ILayer>(layerCollection);
        }

        public event EventHandler<TrainingProgressEventArgs> TrainingProgress;

        public ReadOnlyCollection<ILayer> Layers { get; }

        public void Abort()
        {
            aborted = true;
        }

        public Tensor CalculateOutputs(Tensor inputs)
        {
            var results = inputs;
            for (int l = 0, count = Layers.Count; l < count; l++)
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
                foreach (var layer in Layers)
                {
                    layer.Dump(writer);
                }
            }
        }

        public void Fit(TrainingSet trainingSet, TrainingSet validationSet, int epochs)
        {
            aborted = false;

            var stopwatch = new Stopwatch();
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                if (aborted)
                {
                    break;
                }

                float learningRate = learningPolicy.GetLearningRate(epoch);

                stopwatch.Restart();
                float trainingError = Train(trainingSet, learningRate);
                float validationError = Test(validationSet);
                stopwatch.Stop();

                RaiseTrainingProgress(epoch, trainingError, validationError, learningRate, stopwatch.ElapsedMilliseconds);
            }
        }

        public void Fit(TrainingSet trainingSet, TrainingSet validationSet, TimeSpan timeSpan)
        {
            aborted = false;

            var stopwatch = new Stopwatch();
            var elapsedTime = TimeSpan.Zero;
            var epoch = 0;

            while (elapsedTime < timeSpan && !aborted)
            {
                float learningRate = learningPolicy.GetLearningRate(epoch);

                stopwatch.Restart();
                float trainingError = Train(trainingSet, learningRate);
                float validationError = Test(validationSet);
                stopwatch.Stop();

                elapsedTime += stopwatch.Elapsed;

                RaiseTrainingProgress(epoch++, trainingError, validationError, learningRate, stopwatch.ElapsedMilliseconds);
            }
        }

        public void Restore(string fileName)
        {
            using (var file = File.OpenRead(fileName))
            {
                Restore(file);
            }
        }

        public void Restore(Stream stream)
        {
            using (var reader = new BinaryReader(stream))
            {
                foreach (var layer in Layers)
                {
                    layer.Restore(reader);
                }
            }
        }

        public float Test(TrainingSet validationSet)
        {
            float sumCost = 0f;
            for (int i = 0, count = validationSet.Count; i < count; i++)
            {
                var inputs = validationSet[i].Item1;
                var expected = validationSet[i].Item2;

                var actuals = CalculateOutputs(inputs);

                sumCost += loss.CalculateTotal(actuals, expected);
            }

            return sumCost / validationSet.Count;
        }

        public float Train(TrainingSet trainingSet, float learningRate)
        {
            if (shuffle)
            {
                trainingSet.Shuffle();
            }

            var lastLayer = Layers[Layers.Count - 1];
            var outputs = lastLayer.Outputs;
            var errors = new Tensor(outputs.Dimensions);

            float sumCost = 0f;
            for (int i = 0, count = trainingSet.Count; i < count; i++)
            {
                errors.Clear();

                (var inputs, var expected) = trainingSet[i];

                var actuals = CalculateTrainingOutputs(inputs);

                sumCost += loss.CalculateTotal(actuals, expected);

                for (int o = 0; o < actuals.Length; o++)
                {
                    errors[o] = loss.Derivative(actuals[o], expected[o]);
                }

                BackPropagate(errors, learningRate, momentumRate);
            }

            return sumCost / trainingSet.Count;
        }

        protected void RaiseTrainingProgress(int epoch, float trainingError, float validationError, float learningRate, long cycleTime) => TrainingProgress?.Invoke(this, new TrainingProgressEventArgs(epoch, trainingError, validationError, learningRate, cycleTime));

        private Tensor BackPropagate(Tensor errors, float learningRate, float momentum)
        {
            var signals = errors;
            for (int l = Layers.Count - 1; l > -1; l--)
            {
                signals = Layers[l].Backpropagate(signals, learningRate, momentum);
            }

            return signals;
        }

        private Tensor CalculateTrainingOutputs(Tensor inputs)
        {
            var results = inputs;
            for (int l = 0, count = Layers.Count; l < count; l++)
            {
                results = Layers[l].CalculateOutputs(results, training: true);
            }

            return results;
        }

        public class Builder
        {
            private ILayerBuilder[] layers;
            private ILearningPolicy learningPolicy;
            private ILossFunction loss = new MeanSquareError();
            private float momentumRate;
            private Random seed;
            private bool shuffle = false;

            public NeuralNetwork Build() => new NeuralNetwork(momentumRate, shuffle, seed, learningPolicy, loss, layers);

            public Builder Layers(params ILayerBuilder[] layers)
            {
                this.layers = layers;

                return this;
            }

            public Builder LearningPolicy(ILearningPolicy learningPolicy, float momentumRate)
            {
                this.learningPolicy = learningPolicy;
                this.momentumRate = momentumRate;

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