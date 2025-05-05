namespace TurtleML.Loss;

using System;

/// <summary>
/// Represents the Cross-Entropy Loss function, which measures the difference between two probability distributions.
/// </summary>
/// <remarks>
/// Commonly used for classification tasks. The formula assumes `expected` is a one-hot encoded label and `actual` is a predicted probability.
/// </remarks>
public class CrossEntropyLoss : LossFunctionBase
{
    /// <inheritdoc/>
    public override float Calculate(float actual, float expected) => -expected * (float)Math.Log(actual);
}