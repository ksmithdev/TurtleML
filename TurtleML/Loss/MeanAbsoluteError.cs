namespace TurtleML.Loss;

using System;

/// <summary>
/// Represents the Mean Absolute Error (MAE) loss function, which calculates the average of the absolute differences between predicted and actual values.
/// </summary>
/// <remarks>
/// MAE is robust to outliers compared to MSE. It is often used when large errors should not be heavily penalized.
/// </remarks>
public class MeanAbsoluteError : LossFunctionBase
{
    /// <inheritdoc/>
    public override float Calculate(float actual, float expected) => Math.Abs(actual - expected);
}