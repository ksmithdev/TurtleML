namespace TurtleML;

/// <summary>
/// Represents the loss function used for the calculating the loss of the training network.
/// </summary>
public interface ILossFunction
{
    /// <summary>
    /// Calculates the loss for the supplied actual and expected output values.
    /// </summary>
    /// <param name="actual">The actual output value.</param>
    /// <param name="expected">The expected output value.</param>
    /// <returns>The calculated loss.</returns>
    float Calculate(float actual, float expected);

    /// <summary>
    /// Calculates the loss for the supplied actual and expected output values.
    /// </summary>
    /// <param name="actuals">The collection of actual output values.</param>
    /// <param name="expected">The collection of expected output values.</param>
    /// <returns>The calculated loss.</returns>
    float CalculateTotal(Tensor actuals, Tensor expected);
}