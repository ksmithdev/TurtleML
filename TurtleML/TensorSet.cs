namespace TurtleML;

using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Represents a set of tensors used for training or validation.
/// </summary>
public class TensorSet : List<Tuple<Tensor, Tensor>>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TensorSet"/> class.
    /// </summary>
    /// <param name="collection">The collection of tensor inputs and outputs.</param>
    public TensorSet(IEnumerable<Tuple<Tensor, Tensor>> collection)
        : base(collection)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorSet"/> class.
    /// </summary>
    public TensorSet()
    {
    }

    /// <summary>
    /// Add an input and output to the set.
    /// </summary>
    /// <param name="inputs">The inputs.</param>
    /// <param name="outputs">The outputs.</param>
    public void Add(Tensor inputs, Tensor outputs)
    {
        Add(Tuple.Create(inputs, outputs));
    }

    /// <summary>
    /// Shuffle the order of the set.
    /// </summary>
    public void Shuffle()
    {
        var random = new Random();

        int n = Count;
        while (n > 1)
        {
            int k = random.Next(n--);

            (this[k], this[n]) = (this[n], this[k]);
        }
    }

    /// <summary>
    /// Split the set into two separate sets based on a percent.
    /// </summary>
    /// <param name="percent">The percent to split the set (rounded down).</param>
    /// <returns>The two split sets.</returns>
    public (TensorSet, TensorSet) Split(float percent)
    {
        var offset = (int)(Count * percent);

        return Split(offset);
    }

    /// <summary>
    /// Split the set into two separate sets at a specific index.
    /// </summary>
    /// <param name="index">Te index to split the set.</param>
    /// <returns>The two split sets.</returns>
    public (TensorSet, TensorSet) Split(int index)
    {
        var set1 = new TensorSet(this.Take(index));
        var set2 = new TensorSet(this.Skip(index));

        return (set1, set2);
    }
}