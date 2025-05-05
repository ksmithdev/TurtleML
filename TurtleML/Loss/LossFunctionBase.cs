namespace TurtleML.Loss;

/// <summary>
/// Base class for all loss functions, providing a common interface for calculating errors.
/// </summary>
/// <remarks>
/// Derived classes must implement the `Calculate` method for specific loss calculations. The `CalculateTotal` method aggregates results across tensors.
/// </remarks>
public abstract class LossFunctionBase : ILossFunction
{
    /// <inheritdoc/>
    public abstract float Calculate(float actual, float expected);

    /// <summary>
    /// Computes the total loss by summing individual errors across all elements in the tensors.
    /// </summary>
    /// <param name="actuals">Tensor containing predicted values.</param>
    /// <param name="expected">Tensor containing true labels or targets.</param>
    /// <returns>Total loss value as a float.</returns>
    /// <remarks>
    /// This method iterates through all elements of the tensors and accumulates the error using the `Calculate` method.
    /// </remarks>
    public float CalculateTotal(Tensor actuals, Tensor expected)
    {
        float sumErrorCost = 0f;
        for (int o = 0; o < actuals.Length; o++)
        {
            sumErrorCost += Calculate(actuals[o], expected[o]);
        }

        return sumErrorCost;
    }
}