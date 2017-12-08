using System;

namespace TurtleML
{
    public class TrainingProgressEventArgs : EventArgs
    {
        internal TrainingProgressEventArgs(float trainingError, float validationError, float learningRate, long cycleTime)
        {
            TrainingError = trainingError;
            ValidationError = validationError;
            LearningRate = learningRate;
            CycleTime = cycleTime;
        }

        public long CycleTime { get; }
        public float TrainingError { get; }
        public float ValidationError { get; }
        private float LearningRate { get; }
    }
}