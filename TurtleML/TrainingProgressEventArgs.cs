using System;

namespace TurtleML
{
    public class TrainingProgressEventArgs : EventArgs
    {
        internal TrainingProgressEventArgs(int epoch, float trainingError, float validationError, float learningRate, long cycleTime)
        {
            Epoch = epoch;
            TrainingError = trainingError;
            ValidationError = validationError;
            LearningRate = learningRate;
            CycleTime = cycleTime;
        }

        public long CycleTime { get; }

        public int Epoch { get; }

        public float TrainingError { get; }

        public float ValidationError { get; }

        private float LearningRate { get; }
    }
}