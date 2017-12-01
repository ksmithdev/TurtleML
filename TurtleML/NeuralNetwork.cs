using System.IO;

namespace TurtleML
{
    public class NeuralNetwork
    {
        private readonly ILayer[] layers;
        private readonly bool shuffle;

        private NeuralNetwork(bool shuffle, ILayer[] layers)
        {
            this.shuffle = shuffle;
            this.layers = layers;
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
            float sumErrorCost = 0f;
            for (int i = 0, count = trainingSet.Count; i < count; i++)
            {
                var inputs = trainingSet[i].Item1;
                var expected = trainingSet[i].Item2;

                Tensor actuals = CalculateOutputs(inputs);
                Tensor errors = new Tensor(actuals.Length);

                for (int o = 0; o < actuals.Length; o++)
                {
                    errors[o] = expected[o] - actuals[o];

                    sumErrorCost += 0.5f * errors[o] * errors[o];
                }
            }

            return sumErrorCost / trainingSet.Count;
        }

        public float Train(TrainingSet trainingSet, float learningRate)
        {
            if (shuffle)
                trainingSet.Shuffle();

            float sumErrorCost = 0f;
            for (int i = 0, count = trainingSet.Count; i < count; i++)
            {
                var inputs = trainingSet[i].Item1;
                var expected = trainingSet[i].Item2;

                Tensor actuals = CalculateTrainingOutputs(inputs);
                Tensor errors = new Tensor(actuals.Length);

                for (int o = 0; o < actuals.Length; o++)
                {
                    errors[o] = expected[o] - actuals[o];

                    sumErrorCost += 0.5f * errors[o] * errors[o];
                }

                BackPropagate(errors, learningRate);
            }

            return sumErrorCost / trainingSet.Count;
        }

        private void BackPropagate(Tensor errors, float learningRate)
        {
            ILayer outputLayer = layers[layers.Length - 1];

            outputLayer.Backpropagate(errors, learningRate);
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
            private ILayer[] layers;
            private bool shuffle = false;

            public NeuralNetwork Build()
            {
                return new NeuralNetwork(shuffle, layers);
            }

            public Builder Layers(params ILayerBuilder[] layers)
            {
                this.layers = new ILayer[layers.Length];

                ILayer inputLayer = null;
                for (int l = 0; l < layers.Length; l++)
                    inputLayer = this.layers[l] = layers[l].Build(inputLayer);

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