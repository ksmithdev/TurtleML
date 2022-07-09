namespace TurtleML
{
    using System;
    using System.Diagnostics;
    using System.IO;
    using TurtleML.Layers;
    using TurtleML.LearningPolicies;
    using TurtleML.Loss;

    public sealed class TrainingNetwork : InferenceNetwork
    {
        private readonly ILearningPolicy learningPolicy;

        private readonly float momentumRate;

        private readonly bool shuffle;

        private bool aborted;

        private TrainingNetwork(float momentumRate, bool shuffle, ILearningPolicy learningPolicy, ILossFunction loss, ILayer[] layers)
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

        private float Test(TrainingSet validationSet)
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

        private float Train(TrainingSet trainingSet, float learningRate)
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
                    errors[o] = expected[o] - actuals[o];
                }

                BackPropagate(errors, learningRate, momentumRate);
            }

            return sumCost / trainingSet.Count;
        }

        public class Builder
        {
            private (int width, int heigh, int depth) inputSize;
            private ILayerBuilder[] layers;
            private ILearningPolicy learningPolicy = new FixedLearningPolicy(0.01f);
            private ILossFunction loss = new MeanSquareError();
            private float momentumRate;
            private Random seed;
            private bool shuffle;

            public TrainingNetwork Build()
            {
                var layerCollection = new ILayer[layers.Length];

                ILayer layer = new InputLayer.Builder()
                    .Dimensions(inputSize.width, inputSize.heigh, inputSize.depth)
                    .Build(null);

                for (int l = 0; l < layers.Length; l++)
                {
                    layer = layers[l].Build(layer);
                    layer.Initialize(seed);

                    layerCollection[l] = layer;
                }

                return new TrainingNetwork(momentumRate, shuffle, learningPolicy, loss, layerCollection);
            }

            public Builder InputShape(int width, int height, int depth)
            {
                inputSize = (width, height, depth);

                return this;
            }

            public Builder Layers(params ILayerBuilder[] layers)
            {
                this.layers = layers;

                return this;
            }

            public Builder LearningPolicy(ILearningPolicy learningPolicy, float momentumRate = 0f)
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