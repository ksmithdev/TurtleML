namespace TurtleML
{
    using System;
    using System.Diagnostics;
    using TurtleML.Loss;

    public sealed class TrainableNeuralNetwork : NeuralNetwork
    {
        private readonly ILearningPolicy learningPolicy;

        private readonly float momentumRate;

        private readonly bool shuffle;

        private bool aborted;

        private TrainableNeuralNetwork(float momentumRate, bool shuffle, ILearningPolicy learningPolicy, ILossFunction loss, ILayer[] layers)
            : base(loss, layers)
        {
            this.momentumRate = momentumRate;
            this.shuffle = shuffle;
            this.learningPolicy = learningPolicy;
        }

        public event EventHandler<TrainingProgressEventArgs> TrainingProgress;

        public void Abort()
        {
            aborted = true;
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

        public float Test(TrainingSet validationSet)
        {
            float sumCost = 0f;
            for (int i = 0, count = validationSet.Count; i < count; i++)
            {
                var inputs = validationSet[i].Item1;
                var expected = validationSet[i].Item2;

                var actuals = CalculateOutputs(inputs);

                sumCost += Loss.CalculateTotal(actuals, expected);
            }

            return sumCost / validationSet.Count;
        }

        public float Train(TrainingSet trainingSet, float learningRate)
        {
            if (shuffle)
            {
                trainingSet.Shuffle();
            }

            var lastLayer = Layers[Layers.Length - 1];
            var outputs = lastLayer.Outputs;
            var errors = new Tensor(outputs.Dimensions);

            float sumCost = 0f;
            for (int i = 0, count = trainingSet.Count; i < count; i++)
            {
                errors.Clear();

                (var inputs, var expected) = trainingSet[i];

                var actuals = CalculateTrainingOutputs(inputs);

                sumCost += Loss.CalculateTotal(actuals, expected);

                for (int o = 0; o < actuals.Length; o++)
                {
                    errors[o] = Loss.Derivative(actuals[o], expected[o]);
                }

                BackPropagate(errors, learningRate, momentumRate);
            }

            return sumCost / trainingSet.Count;
        }

        private Tensor BackPropagate(Tensor errors, float learningRate, float momentum)
        {
            var signals = errors;
            for (int l = Layers.Length - 1; l > -1; l--)
            {
                signals = Layers[l].Backpropagate(signals, learningRate, momentum);
            }

            return signals;
        }

        private Tensor CalculateTrainingOutputs(Tensor inputs)
        {
            var results = inputs;
            for (int l = 0, count = Layers.Length; l < count; l++)
            {
                results = Layers[l].CalculateOutputs(results, training: true);
            }

            return results;
        }

        private void RaiseTrainingProgress(int epoch, float trainingError, float validationError, float learningRate, long cycleTime) => TrainingProgress?.Invoke(this, new TrainingProgressEventArgs(epoch, trainingError, validationError, learningRate, cycleTime));

        public class Builder
        {
            private ILayerBuilder[] layers;
            private ILearningPolicy learningPolicy;
            private ILossFunction loss = new MeanSquareError();
            private float momentumRate;
            private Random seed;
            private bool shuffle;

            public TrainableNeuralNetwork Build()
            {
                var layerCollection = new ILayer[layers.Length];

                ILayer inputLayer = null;
                for (int l = 0; l < layers.Length; l++)
                {
                    inputLayer = layers[l].Build(inputLayer);
                    inputLayer.Initialize(seed);

                    layerCollection[l] = inputLayer;
                }

                return new TrainableNeuralNetwork(momentumRate, shuffle, learningPolicy, loss, layerCollection);
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