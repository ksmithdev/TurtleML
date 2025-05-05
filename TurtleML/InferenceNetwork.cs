namespace TurtleML;

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;

/// <summary>
/// Represents a neural network used for inference tasks, composed of layered processing units.
/// </summary>
/// <remarks>
/// The network is initialized with a loss function and a sequence of layers. It supports
/// restoration from binary data and provides methods to calculate outputs through its layers.
/// </remarks>
public class InferenceNetwork
{
    /// <summary>
    /// The collection of layers in the network.
    /// </summary>
    protected readonly ILayer[] Layers;

    /// <summary>
    /// The loss function.
    /// </summary>
    protected readonly ILossFunction Loss;

    /// <summary>
    /// Initializes a new instance of the <see cref="InferenceNetwork"/> class.
    /// </summary>
    /// <param name="loss">The loss function to use for training/evaluation.</param>
    /// <param name="layers">The array of layers composing the network.</param>
    protected InferenceNetwork(ILossFunction loss, ILayer[] layers)
    {
        Loss = loss;
        Layers = layers;
    }

    /// <summary>
    /// Gets the internal layers as an enumerable collection.
    /// </summary>
    public IEnumerable<ILayer> InternalLayers => Layers;

    /// <summary>
    /// Restores a network from a file.
    /// </summary>
    /// <param name="fileInfo">The file containing serialized network data.</param>
    /// <returns>A new <see cref="InferenceNetwork"/> instance.</returns>
    /// <exception cref="InvalidDataException">Thrown if the file contains invalid network data.</exception>
    /// <exception cref="InvalidOperationException">Thrown for version mismatches or invalid types.</exception>
    public static InferenceNetwork Restore(FileInfo fileInfo)
    {
        using var file = fileInfo.OpenRead();
        return Restore(file);
    }

    /// <summary>
    /// Restores a network from a stream.
    /// </summary>
    /// <param name="stream">The stream containing serialized network data.</param>
    /// <returns>A new <see cref="InferenceNetwork"/> instance.</returns>
    /// <exception cref="InvalidDataException">Thrown if the stream contains invalid network data.</exception>
    /// <exception cref="InvalidOperationException">Thrown for version mismatches or invalid types.</exception>
    public static InferenceNetwork Restore(Stream stream)
    {
        using var reader = new BinaryReader(stream);
        var magicNumber = reader.ReadChars(3);
        if (magicNumber[0] != 't' || magicNumber[1] != 'n' || magicNumber[2] != 'n')
        {
            throw new InvalidDataException("Source data is not a valid neural network dump.");
        }

        var version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidOperationException($"Cannot read version {version} format.");
        }

        var lossFunctionType = reader.ReadString();
        if (!(Activator.CreateInstance(Type.GetType(lossFunctionType)) is ILossFunction lossFunction))
        {
            throw new InvalidOperationException($"An invalid loss function \"{lossFunctionType}\" was specified.");
        }

        var layerCount = reader.ReadInt32();
        var layers = new ILayer[layerCount];
        for (int i = 0; i < layerCount; i++)
        {
            var layerType = reader.ReadString();
            if (RuntimeHelpers.GetUninitializedObject(Type.GetType(layerType)) is not ILayer layer)
            {
                throw new InvalidOperationException($"An invalid layer \"{layerType}\" was specified.");
            }

            layer.Restore(reader);
            layers[i] = layer;
        }

        return new InferenceNetwork(lossFunction, layers);
    }

    /// <summary>
    /// Calculates output tensors by processing inputs through all layers.
    /// </summary>
    /// <param name="inputs">The input tensor to process.</param>
    /// <returns>The resulting output tensor after layer transformations.</returns>
    public Tensor CalculateOutputs(Tensor inputs)
    {
        var results = inputs;
        for (int l = 0, count = Layers.Length; l < count; l++)
        {
            results = Layers[l].CalculateOutputs(results, training: false);
        }

        return results;
    }
}