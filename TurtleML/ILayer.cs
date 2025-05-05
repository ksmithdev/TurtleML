namespace TurtleML;

using System;
using System.IO;

/// <summary>
/// Represents an individual layer in a deep learning model.
/// </summary>
public interface ILayer : IOutput
{
    /// <summary>
    /// Performs the backpropagation algorithm, which calculates gradient of the loss function with respect to the weights and biases.
    /// </summary>
    /// <param name="inputs">The input tensor for this layer.</param>
    /// <param name="errors">The error tensor calculated in the previous layer during forward propagation.</param>
    /// <param name="learningRate">The learning rate used to adjust the weights and biases.</param>
    /// <param name="momentumRate">The momentum rate used for updating the weights and biases.</param>
    /// <returns>The updated error tensor for the next layer.</returns>
    Tensor Backpropagate(Tensor inputs, Tensor errors, float learningRate, float momentumRate);

    /// <summary>
    /// Calculates the output of this layer based on the input.
    /// </summary>
    /// <param name="inputs">The input tensor for this layer.</param>
    /// <param name="training">A flag indicating whether the model is in training mode or not.</param>
    /// <returns>The output tensor of this layer.</returns>
    Tensor CalculateOutputs(Tensor inputs, bool training = false);

    /// <summary>
    /// Dumps the weights and biases of this layer into a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer used to dump the data.</param>
    void Dump(BinaryWriter writer);

    /// <summary>
    /// Initializes the weights and biases of this layer using a random number generator.
    /// </summary>
    /// <param name="random">The random number generator used for initialization.</param>
    void Initialize(Random random);

    /// <summary>
    /// Restores the weights and biases of this layer from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader used to restore the data.</param>
    void Restore(BinaryReader reader);
}