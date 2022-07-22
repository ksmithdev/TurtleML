namespace TurtleML;

using System;

/// <summary>
/// Represents the training progress data.
/// </summary>
public class TrainingProgressEventArgs : EventArgs
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TrainingProgressEventArgs"/> class.
    /// </summary>
    /// <param name="epoch">The epoch number.</param>
    /// <param name="trainingError">The training error amount calculated by the loss function.</param>
    /// <param name="validationError">The validation error amount calculated by the loss function.</param>
    /// <param name="learningRate">The learning rate for this epoch.</param>
    /// <param name="cycleTime">The amount of time it took for the epoch.</param>
    internal TrainingProgressEventArgs(int epoch, float trainingError, float validationError, float learningRate, long cycleTime)
    {
        Epoch = epoch;
        TrainingError = trainingError;
        ValidationError = validationError;
        LearningRate = learningRate;
        CycleTime = cycleTime;
    }

    /// <summary>
    /// Gets the cycle time for the training batch in milliseconds.
    /// </summary>
    public long CycleTime { get; }

    /// <summary>
    /// Gets the epoch number for the training batch.
    /// </summary>
    public int Epoch { get; }

    /// <summary>
    /// Gets the learning rate used during the training batch.
    /// </summary>
    public float LearningRate { get; }

    /// <summary>
    /// Gets the average training error for the batch.
    /// </summary>
    public float TrainingError { get; }

    /// <summary>
    /// Gets the average validation error for the batch.
    /// </summary>
    public float ValidationError { get; }
}