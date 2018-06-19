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
        private readonly float momentumRate;
        private readonly Random seed;
        private readonly bool shuffle;
        private bool aborted;

        private NeuralNetwork(float momentumRate, bool shuffle, Random seed, ILearningPolicy learningPolicy, ILossFunction loss, ILayerBuilder[] layers)
        {
            this.momentumRate = momentumRate;
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

        public void Abort()
        {
            aborted = true;
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

        public void Fit(TrainingSet trainingSet, TrainingSet validationSet, int epochs)
        {
            aborted = false;

            var stopwatch = new Stopwatch();
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                if (aborted)
                    break;

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

            while(elapsedTime < timeSpan && !aborted)
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

                sumCost += loss.CalculateCost(actuals, expected);
            }

            return sumCost / trainingSet.Count;
        }

        public float Train(TrainingSet trainingSet, float learningRate)
        {
            if (shuffle)
                trainingSet.Shuffle();

            var lastLayer = layers[layers.Length - 1];
            var outputs = lastLayer.Outputs;
            Tensor errors = new Tensor(outputs.Dimensions);

            float sumCost = 0f;
            for (int i = 0, count = trainingSet.Count; i < count; i++)
            {
                errors.Clear();

                var inputs = trainingSet[i].Item1;
                var expected = trainingSet[i].Item2;

                Tensor actuals = CalculateTrainingOutputs(inputs);

                sumCost += loss.CalculateCost(actuals, expected);

                for (int o = 0; o < actuals.Length; o++)
                    errors[o] = expected[o] - actuals[o];

                BackPropagate(errors, learningRate, momentumRate);
            }

            return sumCost / trainingSet.Count;
        }

        protected void RaiseTrainingProgress(int epoch, float trainingError, float validationError, float learningRate, long cycleTime) => TrainingProgress?.Invoke(this, new TrainingProgressEventArgs(epoch, trainingError, validationError, learningRate, cycleTime));

        private Tensor BackPropagate(Tensor errors, float learningRate, float momentum)
        {
            Tensor signals = errors;
            for (int l = layers.Length - 1; l > -1; l--)
                signals = layers[l].Backpropagate(signals, learningRate, momentum);
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
            private ILearningPolicy learningPolicy;
            private float learningRate;
            private ILossFunction loss = new MeanSquaredError();
            private float momentumRate;
            private Random seed;
            private bool shuffle = false;

            public NeuralNetwork Build()
            {
                return new NeuralNetwork(momentumRate, shuffle, seed, learningPolicy, loss, layers);
            }

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

            [Obsolete("Obsolete. Use LearningPolicy() instead.")]
            public Builder LearningRate(float learningRate, float momentumRate)
            {
                this.learningRate = learningRate;
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