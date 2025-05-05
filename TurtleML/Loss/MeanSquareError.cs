namespace TurtleML.Loss;

/// <summary>
/// Represents the Mean Squared Error (MSE) loss function, which calculates the average of the squared differences between predicted and actual values.
/// </summary>
/// <remarks>
/// MSE is commonly used in regression problems. The 0.5 factor in the calculation simplifies the derivative during gradient descent updates.
/// </remarks>
public class MeanSquareError : LossFunctionBase
{
    /// <inheritdoc/>
    public override float Calculate(float actual, float expected) => 0.5f * (actual - expected) * (actual - expected);
}